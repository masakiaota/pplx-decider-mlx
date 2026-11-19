"""Unmerged Locate LoRA and query-conditioned span scoring for protocol v4."""

import json
import math
import time
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from .protocol import Encoded, Encoder, LengthError, RequestError


class Locate:
    def __init__(self, engine):
        self.engine = engine
        external = getattr(engine.backbone, "locate_merged_path", None)
        folder = Path(external) if external is not None else engine.path / "locate-span"
        self.config = json.loads((folder / "locate_config.json").read_text())
        if self.config["format"] != "pplx-decider-locate-span-v1":
            raise ValueError("Unexpected Locate format")
        self.encoder = Encoder(engine.path, engine.config["codes"], engine.config.get("max_pixels", 1638400))
        tokenizer = self.encoder.tokenizer
        if "<|locate|>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"extra_special_tokens": ["<|locate|>"]}, replace_extra_special_tokens=False)
        self.marker = tokenizer.encode("<|locate|>", add_special_tokens=False)[0]
        if self.marker != self.config["marker_id"]:
            raise ValueError("Locate marker differs from training")
        self.encoder.escapes = sorted(set(self.encoder.escapes + ["<|locate|>"]), key=len, reverse=True)
        model = engine.backbone
        model.lora = mx.load(str(folder / ("marker.safetensors" if external is not None else "adapter.safetensors")))
        model.lora_scale = self.config["lora"]["alpha"] / self.config["lora"]["rank"]
        model.locate_marker = self.marker
        self.head = mx.load(str(folder / "head.safetensors"))
        mx.eval(model.lora, self.head)
        self.merge_seconds = 0.0
        if external is not None:
            self._load_sidecar(folder)
        elif getattr(model, "use_locate_merge", False):
            started = time.perf_counter()
            modules = [key.removesuffix(".lora_a") for key in model.lora if key.endswith(".lora_a")]
            extra = sum(model.lora[name + ".lora_b"].shape[0] * model.lora[name + ".lora_a"].shape[1] * 2 for name in modules)
            if mx.get_active_memory() + extra + 2e9 > engine.memory_limit:
                raise MemoryError("Merged Locate copy exceeds the model memory budget")
            merged = dict(model.weights)
            for name in modules:
                if name + ".scales" in model.weights:
                    raise NotImplementedError("This first merge experiment only supports unquantized Flash weights")
                base = model.weights[name + ".weight"]
                update = model.lora[name + ".lora_b"].astype(mx.float32) @ model.lora[name + ".lora_a"].astype(mx.float32)
                value = (base.astype(mx.float32) + update * model.lora_scale).astype(base.dtype)
                mx.eval(value)
                merged[name + ".weight"] = value
            model.locate_merged_weights = merged
            model.locate_merged_modules = frozenset(modules)
            model.locate_merged_quantization = {}
            self.merge_seconds = time.perf_counter() - started

    def _load_sidecar(self, folder):
        model = self.engine.backbone
        metadata = json.loads((folder / "mlx_config.json").read_text())
        if metadata.get("format") != "decider-locate-merged-mlx-v1" or metadata.get("lora_applied") is not True:
            raise ValueError("Sidecar must explicitly declare merged Locate LoRA")
        base_metadata_path = self.engine.path / "mlx_config.json"
        if base_metadata_path.exists():
            base_revision = json.loads(base_metadata_path.read_text()).get("source_revision")
        else:
            base_revision = json.loads((self.engine.path / "provenance.json").read_text()).get("revision")
        if not base_revision or metadata.get("base_revision") != base_revision:
            raise ValueError("Merged Locate sidecar belongs to a different base revision")
        modules = frozenset(metadata["merged_modules"])
        expected = frozenset(key.removesuffix(".weight") for key, value in model.weights.items()
                             if key.startswith("language_model.layers.") and key.endswith(".weight") and value.ndim == 2)
        if modules != expected:
            raise ValueError("Merged Locate module set differs from the backbone")
        if set(model.lora) != {"language_model.embed_tokens.locate"}:
            raise ValueError("Merged sidecar must contain only the special marker embedding, no LoRA A/B")
        index = json.loads((folder / "model.safetensors.index.json").read_text())
        files = sorted(set(index["weight_map"].values()))
        # Read only small safetensors headers before materializing any sidecar data.
        new_bytes = 0
        for filename in files:
            with (folder / filename).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                if size > 32 * 1024 * 1024:
                    raise ValueError("Unexpectedly large safetensors header")
                header = json.loads(stream.read(size))
            for key, value in header.items():
                if key != "__metadata__":
                    new_bytes += value["data_offsets"][1] - value["data_offsets"][0]
        if mx.get_active_memory() + new_bytes + 2e9 > self.engine.memory_limit:
            raise MemoryError("Merged Locate sidecar exceeds the model memory budget")
        merged = dict(model.weights)
        for name in modules:
            for suffix in ("weight", "scales", "biases"):
                merged.pop(name + "." + suffix, None)
        allowed = {name + "." + suffix for name in modules for suffix in ("weight", "scales", "biases")}
        started = time.perf_counter()
        loaded = set()
        for filename in files:
            values = mx.load(str(folder / filename))
            if not set(values) <= allowed:
                raise ValueError("Unexpected tensor in merged Locate sidecar")
            for key, value in values.items():
                merged[key] = value.astype(model.dtype) if value.dtype in (mx.float16, mx.bfloat16, mx.float32) else value
                loaded.add(key)
            mx.eval([merged[key] for key in values])
        if any(name + ".weight" not in loaded for name in modules):
            raise ValueError("Merged Locate sidecar is missing a linear weight")
        declared_quantized = frozenset(metadata["quantized_modules"])
        actual_quantized = frozenset(name for name in modules if name + ".scales" in loaded)
        if declared_quantized != actual_quantized:
            raise ValueError("Merged Locate quantization metadata differs from tensors")
        model.locate_merged_weights = merged
        model.locate_merged_modules = modules
        model.locate_merged_quantization = metadata["quantization"]
        self.sidecar_bytes = new_bytes
        self.sidecar_load_seconds = time.perf_counter() - started

    def plans(self, text, items):
        tokenizer, encoder = self.encoder.tokenizer, self.encoder
        context = encoder.context_text({"state": text, "questions": {}})
        begin = len(context) - len(encoder.text(text))
        tokens = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
        prefix = tokens["input_ids"] + [encoder.separator_id]
        chunks = []
        for key, question in items:
            instructions = question.get("instructions")
            if instructions is not None and not isinstance(instructions, str):
                raise RequestError("Locate instructions must be text or null")
            instructions = instructions if instructions is not None and instructions.strip() else key
            prompt = "Question:\nLocate verbatim spans:\n" + encoder.text(instructions)
            chunks.append(tokenizer.encode(prompt, add_special_tokens=False) + [self.marker])
        length = len(prefix) + sum(map(len, chunks))
        if self.engine.max_length and length > self.engine.max_length:
            if len(items) == 1:
                raise LengthError(f"Locate input needs {length} tokens, exceeding {self.engine.max_length}; no truncation applied")
            midpoint = len(items) // 2
            return self.plans(text, items[:midpoint]) + self.plans(text, items[midpoint:])
        ids, markers = list(prefix), []
        for chunk in chunks:
            ids.extend(chunk)
            markers.append(len(ids) - 1)
        mapping = tokens["offset_mapping"]
        positions = [i for i, (a, b) in enumerate(mapping) if a < len(context) and b > begin and a < b]
        offsets = [(max(0, mapping[i][0] - begin), min(len(text), mapping[i][1] - begin)) for i in positions]
        encoded = Encoded(np.asarray(ids, np.int32), tuple(key for key, _ in items), (0,) * len(items),
                          (len(prefix), *map(len, chunks)), tuple(markers))
        return [(encoded, positions, offsets)]

    def linear(self, x, name):
        result = x.astype(mx.float32) @ self.head[name + ".weight"].astype(mx.float32).T
        if name + ".bias" in self.head:
            result = result + self.head[name + ".bias"].astype(mx.float32)
        return result

    def spans(self, document, queries):
        ts, te = self.linear(document, "ts"), self.linear(document, "te")
        qs, qe, qp = (self.linear(queries, name) for name in ("qs", "qe", "qp"))
        scale = math.sqrt(self.config["head"]["width"])
        starts = qs @ ts.T / scale + self.head["start_bias"]
        ends = qe @ te.T / scale + self.head["end_bias"]
        thresholds = self.linear(queries, "thresholds")
        count = min(self.config["head"]["proposals"], document.shape[0])
        result = []
        for i in range(queries.shape[0]):
            si, ei = mx.argsort(starts[i])[-count:], mx.argsort(ends[i])[-count:]
            width = ei[None, :] - si[:, None] + 1
            bucket = mx.clip(mx.floor(mx.log2(mx.maximum(width, 1).astype(mx.float32))), 0, 31).astype(mx.int32)
            scores = (ts[si] * mx.tanh(qp[i])) @ te[ei].T / scale
            scores = scores + starts[i, si][:, None] + ends[i, ei][None, :] + self.head["width_bias.weight"][bucket, 0]
            scores = mx.where((width > 0) & (width <= self.config["head"]["max_span"]), scores, -mx.inf)
            probabilities = mx.sigmoid(scores - thresholds[i, 2])
            mx.eval(probabilities, si, ei)
            probability = np.asarray(probabilities)
            rows, cols = np.nonzero(probability >= self.config.get("thresholds", {}).get("default", 0.5))
            a, b = np.asarray(si), np.asarray(ei)
            result.append([(int(a[row]), int(b[col]), float(probability[row, col])) for row, col in zip(rows, cols)])
        return result

    def decide(self, request):
        if request.get("images") or not isinstance(request["state"], str) or not request["state"]:
            raise RequestError("Locate needs nonempty text state and no images")
        if len(request["questions"]) > 128:
            raise RequestError("Locate supports at most 128 questions")
        plans = self.plans(request["state"], list(request["questions"].items()))
        answers = {}
        text = request["state"]
        for encoded, positions, offsets in plans:
            self.engine._preflight(encoded)
            self.engine.backbone.locate_enabled = True
            try:
                hidden = self.engine.backbone.forward(encoded, return_all=True)[0]
                raw = self.spans(hidden[mx.array(positions)], hidden[mx.array(encoded.markers)]) if positions else [[] for _ in encoded.keys]
            finally:
                self.engine.backbone.locate_enabled = False
            for key, spans in zip(encoded.keys, raw):
                unique = {}
                for first, last, score in spans:
                    a, b = offsets[first][0], offsets[last][1]
                    while a < b and text[a].isspace():
                        a += 1
                    while a < b and text[b - 1].isspace():
                        b -= 1
                    if a < b:
                        unique[a, b] = max(unique.get((a, b), 0.0), score)
                answers[key] = {"type": "locate", "spans": [{"start": a, "end": b, "text": text[a:b], "score": p} for (a, b), p in sorted(unique.items())]}
        return {"answers": answers, "usage": {"input_tokens": sum(e.length for e, _, _ in plans), "output_tokens": 0, "sequences": len(plans)}}

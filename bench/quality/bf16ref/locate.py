"""Locate on the BF16 reference: the original rank-64 LoRA added unmerged, and the query-conditioned span head."""
import json
import math

import mlx.core as mx
import numpy as np
from decider_mlx.preprocess import Preprocessor
from decider_mlx.protocol import MARKER, SEPARATOR, SYSTEM, Encoded, LengthError, RequestError, describe

TOKEN = "<|locate|>"


class Locate:
    def __init__(self, engine):
        self.engine = engine
        folder = engine.path / "locate-span"
        self.config = json.loads((folder / "locate_config.json").read_text())
        if self.config["format"] != "pplx-decider-locate-span-v1":
            raise ValueError("Unexpected Locate format")
        self.pre = Preprocessor(engine.path, SYSTEM)   # own tokenizer: it gains the <|locate|> special token
        self.pre.add_special_token(TOKEN)
        self.marker = self.pre.token_id(TOKEN)
        if self.marker != self.config["marker_id"]:
            raise ValueError("Locate marker differs from training")
        self.separator_id = self.pre.token_id(SEPARATOR)
        self.escapes = sorted({*self.pre.special_tokens(), MARKER, SEPARATOR, TOKEN}, key=len, reverse=True)
        model = engine.backbone
        model.lora = mx.load(str(folder / "adapter.safetensors"))
        model.lora_scale = self.config["lora"]["alpha"] / self.config["lora"]["rank"]
        model.locate_marker = self.marker
        self.head = mx.load(str(folder / "head.safetensors"))
        mx.eval(model.lora, self.head)

    def text(self, value):
        text = describe(value)
        for token in self.escapes:
            text = text.replace(token, token.replace("<", "＜").replace(">", "＞"))
        return text

    def plans(self, text, items):
        context = self.pre.chat_context("State:\n" + self.text(text))
        begin = len(context) - len(self.text(text))
        ids, mapping = self.pre.encode_offsets(context)
        prefix = ids + [self.separator_id]
        chunks = []
        for key, question in items:
            instructions = question.get("instructions")
            if instructions is not None and not isinstance(instructions, str):
                raise RequestError("Locate instructions must be text or null")
            instructions = instructions if instructions is not None and instructions.strip() else key
            prompt = "Question:\nLocate verbatim spans:\n" + self.text(instructions)
            chunks.append(self.pre.encode(prompt) + [self.marker])
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
            self.engine.preflight(encoded)
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

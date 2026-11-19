"""The BF16 reference engine: the original checkpoint, unquantized, one sequence at a time.

Tokenization, the chat template, image preprocessing and answers come from decider_mlx (decider_mlx.protocol and
decider_mlx.preprocess, which reproduce transformers 5.17 bit for bit), so the reference and the engine under test
see the same token ids and pixels.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from decider_mlx.protocol import Encoder, RequestError, answer, validate

from .model import Backbone


class Engine:
    def __init__(self, model_path, *, max_length=12000, memory_limit_gb=48):
        self.path = Path(model_path)
        self.max_length = max_length
        self.memory_limit = int(memory_limit_gb * 1e9)
        mx.set_memory_limit(self.memory_limit)
        mx.set_cache_limit(1_000_000_000)
        self.config = json.loads((self.path / "decision_config.json").read_text())
        protocol = json.loads((self.path / "joint_config.json").read_text())
        if protocol.get("version") != 4 or protocol.get("template") != "joint-chunked-readout-v1":
            raise ValueError("The model must use Decider v1.3 protocol v4")
        if self.config.get("attention_mode") != "noncausal_full_attention":
            raise ValueError("Expected the noncausal Decider classifier")
        self.model_name = "pplx-decider-flash-v1.3" if "4B" in self.config.get("base_model", "") else "pplx-decider-v1.3"
        self.temperature = float(self.config["temperature"])
        self.encoder = Encoder(self.path, self.config["codes"], self.config.get("max_pixels", 1638400))
        started = time.perf_counter()
        self.backbone = Backbone(self.path)
        self.load_seconds = time.perf_counter() - started
        self.locate = None
        self.vision = None

    def preflight(self, encoded):
        """Refuse an input whose estimated MLX peak exceeds the memory limit (one layer's live projections,
        chunked recurrence, masks and residuals) instead of swapping; nothing is truncated."""
        config = self.backbone.text
        length, hidden = encoded.length, config["hidden_size"]
        workspace = length * (hidden * 24 + config["intermediate_size"] * 8 + 64 * config["linear_num_value_heads"] * 32)
        if len(encoded.keys) > 1:
            workspace += length * length * 2
        estimate = mx.get_active_memory() + workspace + 1_000_000_000
        if estimate > self.memory_limit:
            raise MemoryError(f"Estimated MLX peak {estimate / 1e9:.2f} GB exceeds {self.memory_limit / 1e9:.2f} GB; input was not truncated")

    def _decision(self, request, mode, temperature, raw=False):
        plans = self.encoder.plan(request, mode, self.max_length)
        result = {}
        for plan in plans:
            self.preflight(plan)
            embeddings = positions = None
            if plan.pixels is not None:
                if self.vision is None:
                    from .vision import Vision
                    self.vision = Vision(self.backbone)
                embeddings, positions = self.vision.inputs(plan)
            logits = self.backbone.forward(plan, embeddings=embeddings, positions=positions)
            mx.eval(logits)
            values = np.asarray(logits)
            if not np.isfinite(values).all():
                raise FloatingPointError("Readout contains non-finite values; no probability response produced")
            for i, (key, count) in enumerate(zip(plan.keys, plan.counts)):
                scores = values[i, :count]
                if raw:
                    result[key] = scores.tolist()
                else:
                    scores = scores / temperature
                    probabilities = np.exp(scores - np.max(scores))
                    result[key] = answer(request["questions"][key], probabilities / probabilities.sum())
        if raw:
            return result
        return {"model": self.model_name, "answers": result,
                "usage": {"input_tokens": sum(p.length for p in plans), "output_tokens": 0, "sequences": len(plans)}}

    def load_locate(self):
        if self.locate is None:
            from .locate import Locate
            self.locate = Locate(self)
        return self.locate

    def decide(self, request, *, execution_mode="separate", temperature=None, return_logits=False):
        """One request -> one response in the official shape (or raw readout logits per question)."""
        scale = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Temperature must be finite and positive")
        if execution_mode not in {"separate", "isolated"}:
            raise ValueError("execution_mode must be separate or isolated")
        validate(request)
        if return_logits and any(q["type"] == "locate" for q in request["questions"].values()):
            raise RequestError("Raw decision logits do not apply to Locate")
        mx.synchronize()
        located = {key: q for key, q in request["questions"].items() if q["type"] == "locate"}
        decisions = {key: q for key, q in request["questions"].items() if q["type"] != "locate"}
        response = self._decision({**request, "questions": decisions}, execution_mode, scale, raw=return_logits) if decisions else {
            "model": self.model_name, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0, "sequences": 0}}
        if located:
            spans = self.load_locate().decide({**request, "questions": located})
            response["answers"].update(spans["answers"])
            response["answers"] = {key: response["answers"][key] for key in request["questions"]}
            response["usage"] = {key: response["usage"][key] + spans["usage"][key] for key in response["usage"]}
        mx.synchronize()
        return response

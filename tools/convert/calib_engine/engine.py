"""Local request interface; each call finishes GPU work before returning."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model import Backbone
from .memory import preflight_weights
from .protocol import Encoder, RequestError, answer


class Engine:
    def __init__(self, model_path, *, execution_mode="separate", mode=None, dtype="bfloat16",
                 max_length=8192, memory_limit_gb=48, eval_layers=True, prune_last=False,
                 share_context=False, delta_backend="reference", _locate_sidecar_preflight=None):
        self.path = Path(model_path)
        self.mode = mode or execution_mode
        if self.mode not in {"separate", "isolated"}:
            raise ValueError("execution_mode must be separate or isolated")
        self.max_length = max_length
        self.memory_limit = int(memory_limit_gb * 1e9)
        if not 0 < self.memory_limit <= 62e9:
            raise ValueError("MLX memory limit must be in (0, 62] GB")
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
        if delta_backend not in {"reference", "native"}:
            raise ValueError("delta_backend must be reference or native")
        self.weight_memory = preflight_weights(self.path, dtype, self.memory_limit, active_bytes=mx.get_active_memory(), sidecar=_locate_sidecar_preflight)
        self.backbone = Backbone(self.path, dtype=dtype, eval_layers=eval_layers, prune_last=prune_last,
                                 share_context=share_context, delta_backend=delta_backend)
        self.load_seconds = time.perf_counter() - started
        self.last_metrics = {}
        self.locate = None
        self.vision = None

    def _preflight(self, encoded):
        # Bound one layer's simultaneously live projections, chunked recurrence, masks and residuals.
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
            self._preflight(plan)
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

    def decide(self, requests=None, questions=None, *, state=None, images=None, temperature=None,
               execution_mode=None, return_logits=False):
        if questions is not None or state is not None:
            requests = {"state": requests if state is None else state, "questions": questions}
            if images is not None:
                requests["images"] = images
        single = isinstance(requests, dict)
        batch = [requests] if single else list(requests)
        scale = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Temperature must be finite and positive")
        mode = execution_mode or self.mode
        for request in batch:
            self.encoder.validate(request)
            if return_logits and any(q["type"] == "locate" for q in request["questions"].values()):
                raise RequestError("Raw decision logits do not apply to Locate")
        mx.synchronize()
        started = time.perf_counter()
        results = []
        for request in batch:
            located = {key: q for key, q in request["questions"].items() if q["type"] == "locate"}
            decisions = {key: q for key, q in request["questions"].items() if q["type"] != "locate"}
            response = self._decision({**request, "questions": decisions}, mode, scale, raw=return_logits) if decisions else {
                "model": self.model_name, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0, "sequences": 0}}
            if located:
                if self.locate is None:
                    from .locate import Locate
                    self.locate = Locate(self)
                spans = self.locate.decide({**request, "questions": located})
                response["answers"].update(spans["answers"])
                response["answers"] = {key: response["answers"][key] for key in request["questions"]}
                response["usage"] = {key: response["usage"][key] + spans["usage"][key] for key in response["usage"]}
            results.append(response)
        mx.synchronize()
        self.last_metrics = {"elapsed_s": time.perf_counter() - started, "active_bytes": mx.get_active_memory(),
                             "peak_bytes": mx.get_peak_memory(), "cache_bytes": mx.get_cache_memory()}
        return results[0] if single else results

    def logits(self, requests, *, execution_mode=None):
        return self.decide(requests, execution_mode=execution_mode, return_logits=True)

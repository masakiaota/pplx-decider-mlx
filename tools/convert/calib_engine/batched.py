"""Padded multi-request forward experiment; each row owns its recurrent state."""
import math
import time

import mlx.core as mx
import numpy as np

from .engine import Engine as SerialEngine
from .model import silu
from .protocol import answer, RequestError


def rotary(model, x, positions):
    c = model.text["rope_parameters"]
    dim = int(model.text["head_dim"] * c.get("partial_rotary_factor", 1.0))
    inv = c["rope_theta"] ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    if positions.ndim == 3:
        frequencies = positions.astype(mx.float32)[..., None] * inv
        section = c.get("mrope_section", [11, 11, 10])
        axes = [1 if i % 3 == 1 and i < section[1] * 3 else 2 if i % 3 == 2 and i < section[2] * 3 else 0 for i in range(dim // 2)]
        angle = mx.stack([frequencies[:, axis, :, i] for i, axis in enumerate(axes)], axis=-1)[:, None]
    else:
        angle = positions[:, None, :, None].astype(mx.float32) * inv
    cos, sin = mx.cos(angle).astype(x.dtype), mx.sin(angle).astype(x.dtype)
    a, b = x[..., :dim // 2], x[..., dim // 2:dim]
    return mx.concatenate([a * cos - b * sin, b * cos + a * sin, x[..., dim:]], axis=-1)


def attention(model, x, name, positions, mask, markers=None):
    c = model.text
    batch, length, _ = x.shape
    heads, kvheads, width = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    rows = mx.arange(batch)[:, None]
    source = x if markers is None else x[rows, markers]
    qg = model.linear(source, name + ".q_proj").reshape(batch, -1, heads, width * 2)
    q, gate = mx.split(qg, 2, axis=-1)
    q = model.norm(q, name + ".q_norm").transpose(0, 2, 1, 3)
    k = model.norm(model.linear(x, name + ".k_proj").reshape(batch, length, kvheads, width), name + ".k_norm").transpose(0, 2, 1, 3)
    v = model.linear(x, name + ".v_proj").reshape(batch, length, kvheads, width).transpose(0, 2, 1, 3)
    if markers is None:
        query_positions = positions
    elif positions.ndim == 3:
        query_positions = positions.transpose(0, 2, 1)[rows, markers].transpose(0, 2, 1)
    else:
        query_positions = positions[rows, markers]
    q = rotary(model, q, query_positions)
    k = rotary(model, k, positions)
    if markers is not None and mask.shape[-2] != 1:
        mask = mask[:, 0][rows, markers][:, None]
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=width ** -0.5, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(batch, source.shape[1], -1) * mx.sigmoid(gate.reshape(batch, source.shape[1], -1))
    return model.linear(out, name + ".o_proj")


def isolated_delta(model, x, name, chunks):
    # Left-pad only the context pass: its zero prefix leaves the zero recurrent
    # state unchanged, so every row's *valid* context ends at the same position.
    context_lengths = [c.segments[0] for c in chunks]
    maximum = max(context_lengths)
    contexts = mx.stack([mx.pad(x[row, :length], [(maximum - length, 0), (0, 0)]) for row, length in enumerate(context_lengths)])
    context_output, state = model.delta(contexts, name, return_state=True)
    question_lengths = [length for chunk in chunks for length in chunk.segments[1:]]
    widest = max(question_lengths)
    blocks, rows, mapping = [], [], []
    for row, chunk in enumerate(chunks):
        offset = chunk.segments[0]
        row_mapping = []
        for size in chunk.segments[1:]:
            row_mapping.append(len(blocks))
            blocks.append(mx.pad(x[row, offset:offset + size], [(0, widest - size), (0, 0)]))
            rows.append(row)
            offset += size
        mapping.append(row_mapping)
    sources = mx.array(rows)
    question_output = model.delta(mx.stack(blocks), name, state=(state[0][sources], state[1][sources]))
    output = []
    for row, chunk in enumerate(chunks):
        parts = [context_output[row, -chunk.segments[0]:]]
        parts.extend(question_output[i, :size] for i, size in zip(mapping[row], chunk.segments[1:]))
        output.append(mx.pad(mx.concatenate(parts), [(0, x.shape[1] - chunk.length), (0, 0)]))
    return mx.stack(output)


def padded_forward_batch(model, chunks, multimodal=None, return_all=False):
    batch, length = len(chunks), max(c.length for c in chunks)
    questions = max(len(c.keys) for c in chunks)
    ids = np.zeros((batch, length), dtype=np.int32)
    positions = np.tile(np.arange(length, dtype=np.int32), (batch, 1))
    markers = np.zeros((batch, questions), dtype=np.int32)
    labels = np.full((batch, length), -3, dtype=np.int32)
    lengths = np.asarray([c.length for c in chunks], np.int32)
    for row, chunk in enumerate(chunks):
        ids[row, :chunk.length] = chunk.ids
        markers[row, :len(chunk.markers)] = chunk.markers
        context = chunk.segments[0]
        labels[row, :context] = -1
        offset = context
        for i, size in enumerate(chunk.segments[1:]):
            labels[row, offset:offset + size] = i
            positions[row, offset:offset + size] = np.arange(size) + context
            offset += size
    x = model.weights["language_model.embed_tokens.weight"][mx.array(ids)]
    if model.locate_enabled:
        x = mx.where((mx.array(ids) == model.locate_marker)[..., None], model.lora["language_model.embed_tokens.locate"].astype(x.dtype), x)
    if multimodal is not None:
        pos3 = np.repeat(positions[:, None, :], 3, axis=1)
        embedded = []
        for row, value in enumerate(multimodal):
            if value is None:
                embedded.append(x[row])
            else:
                image_embeddings, image_positions = value
                embedded.append(mx.pad(image_embeddings[0], [(0, length - chunks[row].length), (0, 0)]))
                pos3[row, :, :chunks[row].length] = np.asarray(image_positions)
        x = mx.stack(embedded)
        positions = pos3
    markers, positions = mx.array(markers), mx.array(positions)
    valid = mx.arange(length)[None, :] < mx.array(lengths)[:, None]
    isolated = any(len(c.keys) > 1 for c in chunks)
    if isolated:
        labels = mx.array(labels)
        mask = ((labels[:, :, None] == -1) | (labels[:, None, :] == -1) | (labels[:, :, None] == labels[:, None, :])) & valid[:, None, :]
        mask = mask[:, None]
    else:
        mask = valid[:, None, None, :]
    interval = model.sync_interval([c.length for c in chunks])
    pruned = False
    for index, kind in enumerate(model.text["layer_types"]):
        prefix = f"language_model.layers.{index}"
        normalized = model.norm(x, prefix + ".input_layernorm")
        if kind == "full_attention":
            selected = markers if model.prune_last and not return_all and index == len(model.text["layer_types"]) - 1 else None
            update = attention(model, normalized, prefix + ".self_attn", positions, mask, selected)
            if selected is not None:
                x = x[mx.arange(batch)[:, None], markers]
                pruned = True
        else:
            update = isolated_delta(model, normalized, prefix + ".linear_attn", chunks) if isolated else model.delta(normalized, prefix + ".linear_attn")
        x = x + update
        normalized = model.norm(x, prefix + ".post_attention_layernorm")
        x = x + model.linear(silu(model.linear(normalized, prefix + ".mlp.gate_proj")) * model.linear(normalized, prefix + ".mlp.up_proj"), prefix + ".mlp.down_proj")
        if interval and (index + 1) % interval == 0:
            mx.eval(x)
    x = model.norm(x, "language_model.norm")
    if return_all:
        return x
    selected = x if pruned else x[mx.arange(batch)[:, None], markers]
    return (selected @ model.readout.T).astype(mx.float32)


def forward_batch(model, chunks, multimodal=None, return_all=False):
    if not model.packed_enabled:
        return padded_forward_batch(model, chunks, multimodal, return_all)
    from .general_packed import forward
    return forward(model, chunks, multimodal, return_all)


class Engine(SerialEngine):
    def __init__(self, *args, batch_size=8, batch_tokens=8192, sort_by_length=False, max_padding_ratio=1.2, packed=True, metal_attention_limit=512, pack_single=False, nax_attention="long", norm_mode="plain", merge_lora=False, locate_merged_path=None, eval_interval=1, image_cache=False, **kwargs):
        if not isinstance(eval_interval, int) or eval_interval < 0:
            raise ValueError("eval_interval must be a nonnegative integer")
        if merge_lora and locate_merged_path is not None:
            raise ValueError("Select runtime merging or a merged sidecar, not both")
        super().__init__(*args, _locate_sidecar_preflight=locate_merged_path, **kwargs)
        self.pack_single = pack_single
        self.backbone.eval_interval = eval_interval
        self.backbone.image_cache_enabled = bool(image_cache)
        self.backbone.locate_merged_path = locate_merged_path
        self.backbone.use_locate_merge = bool(merge_lora or locate_merged_path is not None)
        self.backbone.norm_mode = norm_mode
        self.backbone.nax_attention = nax_attention
        self.backbone.packed_enabled = packed
        self.backbone.metal_attention_limit = metal_attention_limit
        self.batch_size, self.batch_tokens = int(batch_size), int(batch_tokens)
        self.sort_by_length = sort_by_length
        self.max_padding_ratio = float(max_padding_ratio)
        if self.max_padding_ratio < 1:
            raise ValueError("max_padding_ratio must be at least 1")
        if self.batch_size < 1 or self.batch_tokens < 1:
            raise ValueError("batch_size and batch_tokens must be positive")

    def decide(self, requests=None, questions=None, *, state=None, images=None, temperature=None,
               execution_mode=None, return_logits=False):
        if isinstance(requests, dict) and self.pack_single and questions is None and state is None:
            return self.decide([requests], temperature=temperature, execution_mode=execution_mode, return_logits=return_logits)[0]
        if isinstance(requests, dict) or questions is not None or state is not None:
            return super().decide(requests, questions, state=state, images=images, temperature=temperature,
                                  execution_mode=execution_mode, return_logits=return_logits)
        batch = list(requests)
        mode = execution_mode or self.mode
        eligible = all(all(q["type"] != "locate" for q in r["questions"].values())
                       for r in batch)
        if not eligible:
            if return_logits:
                raise RequestError("Raw decision logits do not apply to Locate")
            return self._mixed_batch(batch, mode, temperature)
        scale = self.temperature if temperature is None else float(temperature)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Temperature must be finite and positive")
        mx.synchronize()
        start = time.perf_counter()
        plans = [self.encoder.plan(request, mode, self.max_length) for request in batch]
        visual_features = {}
        pictured = [(i, plan[0]) for i, plan in enumerate(plans) if plan and plan[0].pixels is not None]
        if pictured:
            if self.vision is None:
                from .vision import Vision
                self.vision = Vision(self.backbone)
            # Pack vision projections while retaining image-wise attention. Bound
            # groups by original patch count before any large device allocation.
            vision_groups, current, count = [], [], 0
            for item in pictured:
                n = len(item[1].pixels)
                if current and count + n > 8192:
                    vision_groups.append(current)
                    current, count = [], 0
                current.append(item)
                count += n
            if current:
                vision_groups.append(current)
            for vision_group in vision_groups:
                patches = sum(len(chunk.pixels) for _, chunk in vision_group)
                estimate = patches * self.backbone.config["vision_config"]["hidden_size"] * 40 + mx.get_active_memory() + 1e9
                if estimate > self.memory_limit:
                    raise MemoryError("Packed image workspace exceeds the configured MLX limit")
                pixels = np.concatenate([chunk.pixels for _, chunk in vision_group])
                grids = np.concatenate([chunk.grid for _, chunk in vision_group])
                features = self.vision(pixels, grids)
                offset = 0
                merge = self.backbone.config["vision_config"]["spatial_merge_size"]
                for request_index, chunk in vision_group:
                    n = int(np.prod(chunk.grid, axis=-1).sum()) // (merge * merge)
                    visual_features[request_index] = features[offset:offset + n]
                    offset += n
        pending = [(i, chunk) for i, plan in enumerate(plans) for chunk in plan]
        if self.sort_by_length:
            pending.sort(key=lambda item: item[1].length, reverse=True)
        outputs = [{} for _ in batch]
        groups, group = [], []
        for item in pending:
            maximum = max([c.length for _, c in group] + [item[1].length])
            padded = maximum * (len(group) + 1)
            actual = sum(c.length for _, c in group) + item[1].length
            if group and (len(group) == self.batch_size or padded > self.batch_tokens or (not self.backbone.packed_enabled and padded / actual > self.max_padding_ratio)):
                groups.append(group)
                group = []
            group.append(item)
        if group:
            groups.append(group)
        for group in groups:
            chunks = [chunk for _, chunk in group]
            largest = max(chunks, key=lambda c: c.length)
            c = self.backbone.text
            workspace = len(group) * largest.length * (c["hidden_size"] * 24 + c["intermediate_size"] * 8 + 64 * c["linear_num_value_heads"] * 32)
            if mx.get_active_memory() + workspace + 1e9 > self.memory_limit:
                raise MemoryError("Padded batch workspace exceeds the configured MLX memory limit")
            multimodal = None
            if any(chunk.pixels is not None for chunk in chunks):
                multimodal = [self.vision.inputs(chunk, features=visual_features[index]) if chunk.pixels is not None else None for index, chunk in group]
            logits = forward_batch(self.backbone, chunks, multimodal)
            mx.eval(logits)
            values = np.asarray(logits)
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite batched readout")
            for row, (request_index, chunk) in enumerate(group):
                for question, (key, count) in enumerate(zip(chunk.keys, chunk.counts)):
                    scores = values[row, question, :count]
                    if return_logits:
                        outputs[request_index][key] = scores.tolist()
                    else:
                        scores = scores / scale
                        p = np.exp(scores - np.max(scores))
                        outputs[request_index][key] = answer(batch[request_index]["questions"][key], p / p.sum())
        mx.synchronize()
        self.last_metrics = {"elapsed_s": time.perf_counter() - start, "active_bytes": mx.get_active_memory(),
                             "peak_bytes": mx.get_peak_memory(), "cache_bytes": mx.get_cache_memory(),
                             "batch_forward_sizes": [len(g) for g in groups],
                             "actual_padding_tokens": 0 if self.backbone.packed_enabled else None,
                             "padding_tokens": sum(len(g) * max(c.length for _, c in g) - sum(c.length for _, c in g) for g in groups)}
        if return_logits:
            return outputs
        return [{"model": self.model_name, "answers": {key: outputs[i][key] for key in request["questions"]},
                 "usage": {"input_tokens": sum(c.length for c in plans[i]), "output_tokens": 0, "sequences": len(plans[i])}}
                for i, request in enumerate(batch)]


    def _mixed_batch(self, requests, mode, temperature):
        started = time.perf_counter()
        for request in requests:
            self.encoder.validate(request)
            if any(q["type"] == "locate" for q in request["questions"].values()) and len(request["questions"]) > 128:
                raise RequestError("Locate supports at most 128 questions")
        decisions, owners = [], []
        pending = []
        responses = [{"model": self.model_name, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0, "sequences": 0}} for _ in requests]
        for index, request in enumerate(requests):
            questions = {k: q for k, q in request["questions"].items() if q["type"] != "locate"}
            located = [(k, q) for k, q in request["questions"].items() if q["type"] == "locate"]
            if questions:
                owners.append(index)
                decisions.append({**request, "questions": questions})
            if located:
                if self.locate is None:
                    from .locate import Locate
                    self.locate = Locate(self)
                for plan in self.locate.plans(request["state"], located):
                    pending.append((index, plan))
        decision_metrics = {}
        if decisions:
            values = self.decide(decisions, temperature=temperature, execution_mode=mode)
            decision_metrics = dict(self.last_metrics)
            for index, response in zip(owners, values):
                responses[index] = response
        if self.sort_by_length:
            pending.sort(key=lambda item: item[1][0].length, reverse=True)
        groups, group = [], []
        for item in pending:
            maximum = max([plan[0].length for _, plan in group] + [item[1][0].length])
            padded = maximum * (len(group) + 1)
            actual = sum(plan[0].length for _, plan in group) + item[1][0].length
            if group and (len(group) == self.batch_size or padded > self.batch_tokens or (not self.backbone.packed_enabled and padded / actual > self.max_padding_ratio)):
                groups.append(group)
                group = []
            group.append(item)
        if group:
            groups.append(group)
        self.backbone.locate_enabled = True
        try:
            for group in groups:
                chunks = [plan[0] for _, plan in group]
                c = self.backbone.text
                workspace = len(group) * max(chunk.length for chunk in chunks) * (c["hidden_size"] * 32 + c["intermediate_size"] * 8 + 64 * c["linear_num_value_heads"] * 32)
                if mx.get_active_memory() + workspace + 1e9 > self.memory_limit:
                    raise MemoryError("Locate batch workspace exceeds configured limit")
                hidden = forward_batch(self.backbone, chunks, return_all=True)
                mx.eval(hidden)
                for row, (index, (encoded, positions, offsets)) in enumerate(group):
                    raw = self.locate.spans(hidden[row, mx.array(positions)], hidden[row, mx.array(encoded.markers)]) if positions else [[] for _ in encoded.keys]
                    text = requests[index]["state"]
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
                        responses[index]["answers"][key] = {"type": "locate", "spans": [{"start": a, "end": b, "text": text[a:b], "score": p} for (a, b), p in sorted(unique.items())]}
                    responses[index]["usage"]["input_tokens"] += encoded.length
                    responses[index]["usage"]["sequences"] += 1
        finally:
            self.backbone.locate_enabled = False
        for request, response in zip(requests, responses):
            response["answers"] = {key: response["answers"][key] for key in request["questions"]}
        mx.synchronize()
        self.last_metrics = {"elapsed_s": time.perf_counter() - started, "active_bytes": mx.get_active_memory(),
                             "peak_bytes": mx.get_peak_memory(), "cache_bytes": mx.get_cache_memory(),
                             "batch_forward_sizes": decision_metrics.get("batch_forward_sizes", []),
                             "locate_batch_forward_sizes": [len(group) for group in groups],
                             "actual_padding_tokens": 0 if self.backbone.packed_enabled else None,
                             "padding_tokens": decision_metrics.get("padding_tokens", 0) + sum(len(group) * max(plan[0].length for _, plan in group) - sum(plan[0].length for _, plan in group) for group in groups)}
        return responses

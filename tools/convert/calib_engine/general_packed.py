"""Packed valid tokens with query ranges and forked question recurrence."""
import math

import mlx.core as mx
import numpy as np

from . import metal_ops
from .model import silu


class Layout:
    def __init__(self, chunks, multimodal):
        self.chunks = chunks
        self.lengths = [c.length for c in chunks]
        self.offsets = np.cumsum([0, *self.lengths]).tolist()
        self.starts = mx.array(self.offsets[:-1])
        self.owners = mx.array(np.repeat(np.arange(len(chunks), dtype=np.int32), self.lengths))
        contexts, question_offsets, question_starts, question_ends = [], [0], [], []
        markers, ranges, blocks, prefixes, positions = [], [], [], [], []
        self.groups = []
        for row, chunk in enumerate(chunks):
            start, end = self.offsets[row:row + 2]
            context = start + chunk.segments[0]
            contexts.append(context)
            blocks.extend([start] * chunk.segments[0])
            prefixes.extend([start] * chunk.segments[0])
            ranges.extend([(start, end, 0, 0)] * chunk.segments[0])
            pos = [np.arange(chunk.segments[0])]
            offset = context
            first_marker = len(markers)
            if len(chunk.keys) == 1:
                self.groups.append((start, end, start, end, 0, 0))
            else:
                self.groups.append((start, context, start, end, 0, 0))
            for i, size in enumerate(chunk.segments[1:]):
                question_starts.append(offset)
                question_ends.append(offset + size)
                markers.append(offset + size - 1)
                ranges.extend([(start, context, offset, offset + size)] * size)
                blocks.extend([offset] * size)
                prefixes.extend([context] * size)
                pos.append(np.arange(size) + chunk.segments[0])
                if len(chunk.keys) != 1:
                    self.groups.append((offset, offset + size, start, context, offset, offset + size))
                offset += size
            question_offsets.append(len(question_starts))
            if multimodal is not None and multimodal[row] is not None:
                positions.append(multimodal[row][1])
            else:
                positions.append(mx.array(np.tile(np.concatenate(pos), (3, 1))))
        self.contexts = mx.array(contexts)
        self.question_offsets = mx.array(question_offsets)
        self.question_starts, self.question_ends = mx.array(question_starts), mx.array(question_ends)
        self.markers = mx.array(markers)
        self.ranges = mx.array(ranges)
        self.blocks, self.prefixes = mx.array(blocks), mx.array(prefixes)
        self.positions = mx.concatenate(positions, axis=1)
        self.marker_groups = []
        for row, chunk in enumerate(chunks):
            start = self.offsets[row]
            context, offset = start + chunk.segments[0], start + chunk.segments[0]
            for size in chunk.segments[1:]:
                self.marker_groups.append((start, context, offset, offset + size))
                offset += size


def delta(model, x, name, layout):
    c = model.text
    kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
    kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
    projected = model.linear(x, name + ".in_proj_qkv")
    beta = mx.sigmoid(model.linear(x, name + ".in_proj_b"))
    a = model.linear(x, name + ".in_proj_a").astype(mx.float32) + model.weights[name + ".dt_bias"].astype(mx.float32)
    g = -mx.exp(model.weights[name + ".A_log"].astype(mx.float32)) * mx.logaddexp(a, mx.array(0.0))
    convolved = metal_ops.convolution(projected, model.weights[name + ".conv1d.weight"][:, 0],
                                     layout.starts, layout.owners, layout.blocks, layout.prefixes)
    q, k, v = mx.split(convolved, [kh * kd, 2 * kh * kd], axis=-1)
    q, k = q.reshape(-1, kh, kd).astype(mx.float32), k.reshape(-1, kh, kd).astype(mx.float32)
    v = v.reshape(-1, vh, vd).astype(mx.float32)
    q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6) / math.sqrt(kd)
    k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    out = metal_ops.delta(q, k, v, mx.exp(g), beta.astype(mx.float32), layout.starts, layout.contexts,
                          layout.question_offsets, layout.question_starts, layout.question_ends).astype(x.dtype)
    z = model.linear(x, name + ".in_proj_z").reshape(-1, vh, vd)
    xf = out.astype(mx.float32)
    normalized = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + c["rms_norm_eps"])).astype(x.dtype)
    out = (normalized * model.weights[name + ".norm.weight"] * silu(z.astype(mx.float32))).astype(x.dtype)
    return model.linear(out.reshape(x.shape[0], -1), name + ".out_proj")


def attention(model, x, name, layout, last):
    c = model.text
    heads, kvheads, width = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    source = x[layout.markers] if last else x
    q, gate = mx.split(model.linear(source, name + ".q_proj").reshape(-1, heads, 2 * width), 2, axis=-1)
    q = model.norm(q, name + ".q_norm")
    k = model.norm(model.linear(x, name + ".k_proj").reshape(-1, kvheads, width), name + ".k_norm")
    v = model.linear(x, name + ".v_proj").reshape(-1, kvheads, width)
    q_positions = layout.positions[:, layout.markers] if last else layout.positions
    q = model.rotary(q[None].transpose(0, 2, 1, 3), q_positions).transpose(0, 2, 1, 3)[0]
    k = model.rotary(k[None].transpose(0, 2, 1, 3), layout.positions).transpose(0, 2, 1, 3)[0]
    use_nax = not last and (model.nax_attention == "all" or (model.nax_attention == "long" and max(layout.lengths) > model.metal_attention_limit))
    if use_nax:
        from .nax_attention import attention as tiled_attention
        key_parts, value_parts, q_lengths, k_lengths = [], [], [], []
        for qa, qb, ka, kb, kc, kd in layout.groups:
            key_parts.append(k[ka:kb] if kc == kd else mx.concatenate([k[ka:kb], k[kc:kd]]))
            value_parts.append(v[ka:kb] if kc == kd else mx.concatenate([v[ka:kb], v[kc:kd]]))
            q_lengths.append(qb - qa)
            k_lengths.append(kb - ka + kd - kc)
        out = tiled_attention(q, mx.concatenate(key_parts), mx.concatenate(value_parts), q_lengths, k_lengths)
    elif max(layout.lengths) <= model.metal_attention_limit:
        ranges = layout.ranges[layout.markers] if last else layout.ranges
        out = metal_ops.attention(q, k, v, ranges)
    else:
        parts = []
        groups = [(i, i + 1, *group) for i, group in enumerate(layout.marker_groups)] if last else layout.groups
        for qa, qb, ka, kb, kc, kd in groups:
            query = q[qa:qb][None].transpose(0, 2, 1, 3)
            keys = k[ka:kb] if kc == kd else mx.concatenate([k[ka:kb], k[kc:kd]])
            values = v[ka:kb] if kc == kd else mx.concatenate([v[ka:kb], v[kc:kd]])
            out = mx.fast.scaled_dot_product_attention(query, keys[None].transpose(0, 2, 1, 3),
                                                       values[None].transpose(0, 2, 1, 3), scale=width ** -0.5)
            parts.append(out.transpose(0, 2, 1, 3)[0])
        out = mx.concatenate(parts)
    out = out.reshape(source.shape[0], -1) * mx.sigmoid(gate.reshape(source.shape[0], -1))
    return model.linear(out, name + ".o_proj")


def forward(model, chunks, multimodal=None, return_all=False):
    layout = Layout(chunks, multimodal)
    ids = mx.array(np.concatenate([c.ids for c in chunks]))
    if multimodal is None:
        x = model.weights["language_model.embed_tokens.weight"][ids]
    else:
        x = mx.concatenate([model.weights["language_model.embed_tokens.weight"][mx.array(chunk.ids)] if value is None else value[0][0]
                            for chunk, value in zip(chunks, multimodal)])
    if model.locate_enabled:
        x = mx.where((ids == model.locate_marker)[:, None], model.lora["language_model.embed_tokens.locate"].astype(x.dtype), x)
    interval = model.sync_interval(layout.lengths)
    pruned = False
    for index, kind in enumerate(model.text["layer_types"]):
        prefix = f"language_model.layers.{index}"
        normalized = model.norm(x, prefix + ".input_layernorm")
        last = model.prune_last and not return_all and index == len(model.text["layer_types"]) - 1
        if kind == "full_attention":
            update = attention(model, normalized, prefix + ".self_attn", layout, last)
            if last:
                x = x[layout.markers]
                pruned = True
        else:
            update = delta(model, normalized, prefix + ".linear_attn", layout)
        x = x + update
        normalized = model.norm(x, prefix + ".post_attention_layernorm")
        x = x + model.linear(silu(model.linear(normalized, prefix + ".mlp.gate_proj")) * model.linear(normalized, prefix + ".mlp.up_proj"), prefix + ".mlp.down_proj")
        if interval and (index + 1) % interval == 0:
            mx.eval(x)
    x = model.norm(x, "language_model.norm")
    if return_all:
        maximum = max(layout.lengths)
        return mx.stack([mx.pad(x[a:b], [(0, maximum - (b - a)), (0, 0)]) for a, b in zip(layout.offsets[:-1], layout.offsets[1:])])
    logits = ((x if pruned else x[layout.markers]) @ model.readout.T).astype(mx.float32)
    maximum, result, offset = max(len(c.keys) for c in chunks), [], 0
    for chunk in chunks:
        count = len(chunk.keys)
        result.append(mx.pad(logits[offset:offset + count], [(0, maximum - count), (0, 0)]))
        offset += count
    return mx.stack(result)

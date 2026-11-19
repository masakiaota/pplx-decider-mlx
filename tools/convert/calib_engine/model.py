"""Native MLX computations for the two Decider v1.3 dense backbones."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx


def rms(x, weight, eps=1e-6, zero_centered=True):
    xf = x.astype(mx.float32)
    w = weight.astype(mx.float32) + (1.0 if zero_centered else 0.0)
    return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps) * w).astype(x.dtype)


compiled_rms = mx.compile(rms)


def silu(x):
    return x * mx.sigmoid(x)


def delta_chunk(q, k, v, g, beta, chunk_size=64, initial_state=None):
    """FP32 gated delta rule; logarithmic nilpotent inverse of each triangular chunk."""
    dtype = q.dtype
    batch, length, heads, width = v.shape
    if q.shape[2] != heads:
        repeats = heads // q.shape[2]
        q, k = mx.repeat(q, repeats, axis=2), mx.repeat(k, repeats, axis=2)
    q, k, v = (x.transpose(0, 2, 1, 3).astype(mx.float32) for x in (q, k, v))
    beta, g = (x.transpose(0, 2, 1).astype(mx.float32) for x in (beta, g))
    q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6) / math.sqrt(q.shape[-1])
    k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    pad = (-length) % chunk_size
    if pad:
        q, k, v = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (q, k, v))
        beta, g = (mx.pad(x, [(0, 0), (0, 0), (0, pad)]) for x in (beta, g))
    count = (length + pad) // chunk_size
    q, k, v = (x.reshape(batch, heads, count, chunk_size, -1) for x in (q, k, v))
    beta = beta.reshape(batch, heads, count, chunk_size, 1)
    cumulative = mx.cumsum(g.reshape(batch, heads, count, chunk_size), axis=-1)
    row = mx.arange(chunk_size)[:, None]
    col = mx.arange(chunk_size)[None, :]
    decay = mx.exp(mx.where(row >= col, cumulative[..., :, None] - cumulative[..., None, :], -mx.inf))
    a = -mx.where(row > col, (k * beta) @ k.swapaxes(-1, -2) * decay, 0)
    # Stable forward substitution. A nilpotent power-series inverse is unsafe for
    # the strongly correlated keys in trained weights (large cancelling terms).
    for i in range(1, chunk_size):
        row_value = a[..., i, :i]
        update = row_value + mx.sum(row_value[..., :, None] * a[..., :i, :i], axis=-2)
        a[..., i, :i] = update
    inv = a + mx.eye(chunk_size)
    values = inv @ (v * beta)
    corrected_keys = inv @ (k * beta * mx.exp(cumulative)[..., None])
    within = (q @ k.swapaxes(-1, -2)) * decay
    q = q * mx.exp(cumulative)[..., None]
    k = k * mx.exp(cumulative[..., -1:] - cumulative)[..., None]
    state = mx.zeros((batch, heads, q.shape[-1], width), mx.float32) if initial_state is None else initial_state
    outputs = []
    for i in range(count):
        update = values[:, :, i] - corrected_keys[:, :, i] @ state
        outputs.append(q[:, :, i] @ state + within[:, :, i] @ update)
        state = state * mx.exp(cumulative[:, :, i, -1])[..., None, None] + k[:, :, i].swapaxes(-1, -2) @ update
    output = mx.stack(outputs, axis=2).reshape(batch, heads, length + pad, width)[:, :, :length]
    return output.transpose(0, 2, 1, 3).astype(dtype), state


class Backbone:
    def __init__(self, directory, *, dtype="bfloat16", eval_layers=True, prune_last=False, share_context=False, delta_backend="reference"):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        self.text = self.config["text_config"]
        if self.config["model_type"] != "qwen3_5":
            raise ValueError("Only the two dense Decider v1.3 backbones are supported")
        self.dtype = getattr(mx, dtype)
        self.eval_layers = eval_layers
        self.eval_interval = 1
        self.prune_last = prune_last
        self.share_context = share_context
        self.delta_backend = delta_backend
        self.weights = {}
        self.lora = {}
        self.lora_scale = 1.0
        self.locate_enabled = False
        self.quantization = {}
        metadata = self.directory / "mlx_config.json"
        if metadata.exists():
            self.quantization = json.loads(metadata.read_text()).get("quantization", {})
        index_path = self.directory / "model.safetensors.index.json"
        if index_path.exists():
            files = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        else:
            files = [p.name for p in self.directory.glob("model*.safetensors")]
        if not files:
            raise FileNotFoundError("No model weight shards found")
        self.files = files
        for filename in files:
            shard = mx.load(str(self.directory / filename))
            for name, value in shard.items():
                if name.startswith("language_model."):
                    self.weights[name] = value.astype(self.dtype) if value.dtype in (mx.float16, mx.bfloat16, mx.float32) else value
            mx.eval(self.weights)
        self.readout = mx.load(str(self.directory / "readout.safetensors"))["weight"].astype(self.dtype)
        mx.eval(self.readout)

    def sync_interval(self, lengths):
        # Final-only scheduling has been measured only in this bounded range.
        # Longer/larger groups keep layer-wise evaluation instead of rejecting input.
        interval = self.eval_interval
        if interval != 1 and (len(lengths) > 8 or max(lengths, default=0) > 5500 or sum(lengths) > 8192):
            interval = 1
        self.last_sync_interval = interval
        return interval

    def load_vision(self):
        if "visual.patch_embed.proj.weight" in self.weights:
            return
        for filename in self.files:
            shard = mx.load(str(self.directory / filename))
            for name, value in shard.items():
                if name.startswith("visual."):
                    self.weights[name] = value.astype(self.dtype)
            mx.eval(self.weights)

    def linear(self, x, name):
        merged = self.locate_enabled and getattr(self, "use_locate_merge", False) and name in self.locate_merged_modules
        weights = self.locate_merged_weights if merged else self.weights
        weight = weights[name + ".weight"]
        if name + ".scales" in weights:
            quantization = self.locate_merged_quantization if merged else self.quantization
            config = quantization.get(name, quantization)
            result = mx.quantized_matmul(x, weight, weights[name + ".scales"], weights.get(name + ".biases"),
                                         transpose=True, group_size=config.get("group_size", 64), bits=config.get("bits", 4), mode=config.get("mode", "affine"))
        else:
            result = x @ weight.T
        if name + ".bias" in weights:
            result = result + weights[name + ".bias"]
        if self.locate_enabled and not merged and name + ".lora_a" in self.lora:
            update = (x.astype(mx.float32) @ self.lora[name + ".lora_a"].astype(mx.float32).T) @ self.lora[name + ".lora_b"].astype(mx.float32).T
            result = result + (update * self.lora_scale).astype(result.dtype)
        return result

    def norm(self, x, name, zero_centered=True):
        mode = getattr(self, "norm_mode", "plain")
        if mode == "compiled":
            return compiled_rms(x, self.weights[name + ".weight"], self.text["rms_norm_eps"], zero_centered)
        if mode == "native":
            if not hasattr(self, "_norm_cache"):
                self._norm_cache = {}
            key = (name, zero_centered)
            if key not in self._norm_cache:
                self._norm_cache[key] = self.weights[name + ".weight"].astype(mx.float32) + (1.0 if zero_centered else 0.0)
                mx.eval(self._norm_cache[key])
            return mx.fast.rms_norm(x.astype(mx.float32), self._norm_cache[key], self.text["rms_norm_eps"]).astype(x.dtype)
        return rms(x, self.weights[name + ".weight"], self.text["rms_norm_eps"], zero_centered)

    def rotary(self, x, positions):
        config = self.text["rope_parameters"]
        dim = int(self.text["head_dim"] * config.get("partial_rotary_factor", 1.0))
        inv = config["rope_theta"] ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        if positions.ndim == 1:
            angle = positions[:, None].astype(mx.float32) * inv
        else:
            angles = positions[..., None].astype(mx.float32) * inv
            indices = list(range(dim // 2))
            section = config.get("mrope_section", [11, 11, 10])
            dimensions = [1 if i % 3 == 1 and i < section[1] * 3 else 2 if i % 3 == 2 and i < section[2] * 3 else 0 for i in indices]
            angle = mx.stack([angles[d, :, i] for i, d in enumerate(dimensions)], axis=-1)
        cos, sin = mx.cos(angle).astype(x.dtype)[None, None], mx.sin(angle).astype(x.dtype)[None, None]
        first, second = x[..., :dim // 2], x[..., dim // 2:dim]
        return mx.concatenate([first * cos - second * sin, second * cos + first * sin, x[..., dim:]], axis=-1)

    def attention(self, x, name, positions, mask=None, selected=None):
        c = self.text
        batch, length, _ = x.shape
        heads, kv_heads, width = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        source = x if selected is None else x[:, selected]
        qg = self.linear(source, name + ".q_proj").reshape(batch, -1, heads, width * 2)
        q, gate = mx.split(qg, 2, axis=-1)
        q = self.norm(q, name + ".q_norm").transpose(0, 2, 1, 3)
        k = self.norm(self.linear(x, name + ".k_proj").reshape(batch, length, kv_heads, width), name + ".k_norm").transpose(0, 2, 1, 3)
        v = self.linear(x, name + ".v_proj").reshape(batch, length, kv_heads, width).transpose(0, 2, 1, 3)
        qpos = positions if selected is None else positions[..., selected]
        q, k = self.rotary(q, qpos), self.rotary(k, positions)
        if selected is not None and mask is not None:
            mask = mask[..., selected, :]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=width ** -0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, source.shape[1], -1)
        out = out * mx.sigmoid(gate.reshape(batch, source.shape[1], -1))
        return self.linear(out, name + ".o_proj")

    def delta(self, x, name, state=None, return_state=False):
        c = self.text
        batch, length, _ = x.shape
        kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
        qkv = self.linear(x, name + ".in_proj_qkv")
        kernel = self.weights[name + ".conv1d.weight"][:, 0, :]
        size = kernel.shape[-1]
        previous = mx.zeros((batch, size - 1, qkv.shape[-1]), qkv.dtype) if state is None else state[0]
        padded = mx.concatenate([previous, qkv], axis=1).astype(mx.float32)
        conv = sum(padded[:, i:i + length] * kernel[:, i].astype(mx.float32) for i in range(size)).astype(qkv.dtype)
        qkv = silu(conv)
        q, k, v = mx.split(qkv, [kh * kd, 2 * kh * kd], axis=-1)
        q, k = q.reshape(batch, length, kh, kd), k.reshape(batch, length, kh, kd)
        v = v.reshape(batch, length, vh, vd)
        beta = mx.sigmoid(self.linear(x, name + ".in_proj_b"))
        a = self.linear(x, name + ".in_proj_a").astype(mx.float32) + self.weights[name + ".dt_bias"].astype(mx.float32)
        g = -mx.exp(self.weights[name + ".A_log"].astype(mx.float32)) * mx.logaddexp(a, mx.array(0.0))
        recurrent = None if state is None else state[1]
        if self.delta_backend == "native":
            qf, kf = q.astype(mx.float32), k.astype(mx.float32)
            qf = qf * mx.rsqrt(mx.sum(qf * qf, axis=-1, keepdims=True) + 1e-6) / math.sqrt(kd)
            kf = kf * mx.rsqrt(mx.sum(kf * kf, axis=-1, keepdims=True) + 1e-6)
            out, final_state = mx.fast.gated_delta_update(qf, kf, v.astype(mx.float32), mx.exp(g), beta.astype(mx.float32), initial_state=recurrent)
            out = out.astype(q.dtype)
        else:
            out, final_state = delta_chunk(q, k, v, g, beta, initial_state=recurrent)
        z = self.linear(x, name + ".in_proj_z").reshape(batch, length, vh, vd)
        # The gated normalization rounds to the model dtype before multiplying its learned weight.
        xf = out.astype(mx.float32)
        normalized = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + c["rms_norm_eps"])).astype(out.dtype)
        out = (normalized * self.weights[name + ".norm.weight"] * silu(z.astype(mx.float32))).astype(x.dtype)
        result = self.linear(out.reshape(batch, length, -1), name + ".out_proj")
        return (result, (padded[:, -(size - 1):].astype(qkv.dtype), final_state)) if return_state else result

    def forward(self, encoded, *, embeddings=None, positions=None, return_all=False):
        ids = mx.array(encoded.ids)[None]
        x = self.weights["language_model.embed_tokens.weight"][ids] if embeddings is None else embeddings
        if self.locate_enabled and "language_model.embed_tokens.locate" in self.lora:
            x = mx.where((ids == self.locate_marker)[..., None], self.lora["language_model.embed_tokens.locate"].astype(x.dtype), x)
        length = encoded.length
        context = encoded.segments[0]
        if positions is None:
            positions = mx.arange(length)
            if len(encoded.keys) > 1:
                positions = mx.concatenate([mx.arange(context), *(mx.arange(size) + context for size in encoded.segments[1:])])
        mask = None
        if len(encoded.keys) > 1:
            labels = mx.concatenate([mx.full((context,), -1), *(mx.full((size,), i) for i, size in enumerate(encoded.segments[1:]))])
            mask = ((labels[:, None] == -1) | (labels[None, :] == -1) | (labels[:, None] == labels[None, :]))[None, None]
        interval = self.sync_interval([encoded.length])
        for index, kind in enumerate(self.text["layer_types"]):
            prefix = f"language_model.layers.{index}"
            normalized = self.norm(x, prefix + ".input_layernorm")
            selected = mx.array(encoded.markers) if self.prune_last and not return_all and index == len(self.text["layer_types"]) - 1 and kind == "full_attention" else None
            if kind == "full_attention":
                update = self.attention(normalized, prefix + ".self_attn", positions, mask, selected)
                if selected is not None:
                    x = x[:, selected]
            else:
                if len(encoded.keys) == 1:
                    update = self.delta(normalized, prefix + ".linear_attn")
                elif self.share_context:
                    first, state = self.delta(normalized[:, :context], prefix + ".linear_attn", return_state=True)
                    blocks, offset = [first], context
                    for size in encoded.segments[1:]:
                        blocks.append(self.delta(normalized[:, offset:offset + size], prefix + ".linear_attn", state=state))
                        offset += size
                    update = mx.concatenate(blocks, axis=1)
                else:
                    # Exact baseline restart: each block sees the layer's same context and its own tokens.
                    blocks, offset = [], context
                    for question, size in enumerate(encoded.segments[1:]):
                        part = mx.concatenate([normalized[:, :context], normalized[:, offset:offset + size]], axis=1)
                        value = self.delta(part, prefix + ".linear_attn")
                        blocks.append(value if question == 0 else value[:, context:])
                        offset += size
                    update = mx.concatenate(blocks, axis=1)
            x = x + update
            normalized = self.norm(x, prefix + ".post_attention_layernorm")
            x = x + self.linear(silu(self.linear(normalized, prefix + ".mlp.gate_proj")) * self.linear(normalized, prefix + ".mlp.up_proj"), prefix + ".mlp.down_proj")
            if self.eval_layers and interval and (index + 1) % interval == 0:
                mx.eval(x)
        x = self.norm(x, "language_model.norm")
        if return_all:
            return x
        if not self.prune_last:
            x = x[:, mx.array(encoded.markers)]
        return (x @ self.readout.T).astype(mx.float32)[0]

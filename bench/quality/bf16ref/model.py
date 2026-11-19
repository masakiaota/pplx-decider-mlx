"""Plain MLX forward of the two Decider v1.3 dense backbones in BF16 (the reference for quality measurements).

Written for clarity, not speed: every linear layer is a BF16 matmul, the Gated DeltaNet layers use a chunked FP32
reference of the gated delta rule (no custom kernels), and every layer is evaluated before the next one. The Locate
LoRA is added unmerged in FP32 when Locate is enabled.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx


def rms(x, weight, eps=1e-6, zero_centered=True):
    xf = x.astype(mx.float32)
    w = weight.astype(mx.float32) + (1.0 if zero_centered else 0.0)
    return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps) * w).astype(x.dtype)


def silu(x):
    return x * mx.sigmoid(x)


def delta_chunk(q, k, v, g, beta, chunk_size=64, initial_state=None):
    """FP32 gated delta rule over chunks of 64 tokens; the triangular system of each chunk is solved by forward
    substitution."""
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
    # Stable forward substitution. A nilpotent power-series inverse is unsafe for the strongly correlated keys in
    # trained weights (large cancelling terms).
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
    def __init__(self, directory, dtype=mx.bfloat16):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        self.text = self.config["text_config"]
        if self.config["model_type"] != "qwen3_5":
            raise ValueError("Only the two dense Decider v1.3 backbones are supported")
        self.dtype = dtype
        self.weights = {}
        self.lora = {}
        self.lora_scale = 1.0
        self.locate_enabled = False
        self.locate_marker = None
        index_path = self.directory / "model.safetensors.index.json"
        if index_path.exists():
            files = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        else:
            files = [p.name for p in self.directory.glob("model*.safetensors")]
        if not files:
            raise FileNotFoundError(f"No model weight shards in {self.directory}")
        self.files = files
        for filename in files:
            shard = mx.load(str(self.directory / filename))
            for name, value in shard.items():
                if name.startswith("language_model."):
                    self.weights[name] = value.astype(self.dtype) if value.dtype in (mx.float16, mx.bfloat16, mx.float32) else value
            mx.eval(self.weights)
        self.readout = mx.load(str(self.directory / "readout.safetensors"))["weight"].astype(self.dtype)
        mx.eval(self.readout)

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
        result = x @ self.weights[name + ".weight"].T
        if name + ".bias" in self.weights:
            result = result + self.weights[name + ".bias"]
        if self.locate_enabled and name + ".lora_a" in self.lora:
            update = (x.astype(mx.float32) @ self.lora[name + ".lora_a"].astype(mx.float32).T) @ self.lora[name + ".lora_b"].astype(mx.float32).T
            result = result + (update * self.lora_scale).astype(result.dtype)
        return result

    def norm(self, x, name, zero_centered=True):
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

    def attention(self, x, name, positions, mask=None):
        c = self.text
        batch, length, _ = x.shape
        heads, kv_heads, width = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        qg = self.linear(x, name + ".q_proj").reshape(batch, -1, heads, width * 2)
        q, gate = mx.split(qg, 2, axis=-1)
        q = self.norm(q, name + ".q_norm").transpose(0, 2, 1, 3)
        k = self.norm(self.linear(x, name + ".k_proj").reshape(batch, length, kv_heads, width), name + ".k_norm").transpose(0, 2, 1, 3)
        v = self.linear(x, name + ".v_proj").reshape(batch, length, kv_heads, width).transpose(0, 2, 1, 3)
        q, k = self.rotary(q, positions), self.rotary(k, positions)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=width ** -0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        out = out * mx.sigmoid(gate.reshape(batch, length, -1))
        return self.linear(out, name + ".o_proj")

    def delta(self, x, name):
        c = self.text
        batch, length, _ = x.shape
        kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
        qkv = self.linear(x, name + ".in_proj_qkv")
        kernel = self.weights[name + ".conv1d.weight"][:, 0, :]
        size = kernel.shape[-1]
        previous = mx.zeros((batch, size - 1, qkv.shape[-1]), qkv.dtype)
        padded = mx.concatenate([previous, qkv], axis=1).astype(mx.float32)
        conv = sum(padded[:, i:i + length] * kernel[:, i].astype(mx.float32) for i in range(size)).astype(qkv.dtype)
        qkv = silu(conv)
        q, k, v = mx.split(qkv, [kh * kd, 2 * kh * kd], axis=-1)
        q, k = q.reshape(batch, length, kh, kd), k.reshape(batch, length, kh, kd)
        v = v.reshape(batch, length, vh, vd)
        beta = mx.sigmoid(self.linear(x, name + ".in_proj_b"))
        a = self.linear(x, name + ".in_proj_a").astype(mx.float32) + self.weights[name + ".dt_bias"].astype(mx.float32)
        g = -mx.exp(self.weights[name + ".A_log"].astype(mx.float32)) * mx.logaddexp(a, mx.array(0.0))
        out, _ = delta_chunk(q, k, v, g, beta)
        z = self.linear(x, name + ".in_proj_z").reshape(batch, length, vh, vd)
        # The gated normalization rounds to the model dtype before multiplying its learned weight.
        xf = out.astype(mx.float32)
        normalized = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + c["rms_norm_eps"])).astype(out.dtype)
        out = (normalized * self.weights[name + ".norm.weight"] * silu(z.astype(mx.float32))).astype(x.dtype)
        return self.linear(out.reshape(batch, length, -1), name + ".out_proj")

    def forward(self, encoded, *, embeddings=None, positions=None, return_all=False):
        """Readout logits (questions, codes) in FP32, or the final hidden states (1, T, D) with return_all."""
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
            # isolated mode: every question block sees the context and itself
            labels = mx.concatenate([mx.full((context,), -1), *(mx.full((size,), i) for i, size in enumerate(encoded.segments[1:]))])
            mask = ((labels[:, None] == -1) | (labels[None, :] == -1) | (labels[:, None] == labels[None, :]))[None, None]
        for index, kind in enumerate(self.text["layer_types"]):
            prefix = f"language_model.layers.{index}"
            normalized = self.norm(x, prefix + ".input_layernorm")
            if kind == "full_attention":
                update = self.attention(normalized, prefix + ".self_attn", positions, mask)
            elif len(encoded.keys) == 1:
                update = self.delta(normalized, prefix + ".linear_attn")
            else:
                # isolated mode: each question block restarts the recurrence on the same context
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
            mx.eval(x)
        x = self.norm(x, "language_model.norm")
        if return_all:
            return x
        x = x[:, mx.array(encoded.markers)]
        return (x @ self.readout.T).astype(mx.float32)[0]

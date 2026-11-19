"""Qwen3.5 image encoder and multi-axis (M-RoPE) positions for Decider v1.3, in plain MLX (BF16 reference)."""

import math

import mlx.core as mx
import numpy as np


def gelu(x, approximate=False):
    if approximate:
        return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)))
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


def layout(grid, side, merge):
    coords, gather, weights, lengths = [], [], [], []
    for temporal, height, width in np.asarray(grid).tolist():
        for _ in range(temporal):
            lengths.append(height * width)
            for block_h in range(height // merge):
                for block_w in range(width // merge):
                    for dh in range(merge):
                        for dw in range(merge):
                            h, w = block_h * merge + dh, block_w * merge + dw
                            coords.append((h, w))
                            rh, rw = h * (side - 1) / max(height - 1, 1), w * (side - 1) / max(width - 1, 1)
                            h0, w0 = math.floor(rh), math.floor(rw)
                            gather.append([min(h0 + a, side - 1) * side + min(w0 + b, side - 1) for a in (0, 1) for b in (0, 1)])
                            weights.append([(1 - abs(rh - h0 - a)) * (1 - abs(rw - w0 - b)) for a in (0, 1) for b in (0, 1)])
    return mx.array(coords), mx.array(gather), mx.array(weights, mx.float32), lengths


class Vision:
    def __init__(self, backbone):
        self.model = backbone
        self.config = backbone.config["vision_config"]

    def norm(self, x, name):
        w = self.model.weights
        return mx.fast.layer_norm(x, w[name + ".weight"], w[name + ".bias"], eps=1e-6)

    def __call__(self, pixels, grid):
        self.model.load_vision()
        c = self.config
        w = self.model.weights
        width = c["hidden_size"] // c["num_heads"]
        coords, indices, interp, lengths = layout(grid, int(math.sqrt(c["num_position_embeddings"])), c["spatial_merge_size"])
        x = mx.array(pixels).astype(self.model.dtype) @ w["visual.patch_embed.proj.weight"].reshape(c["hidden_size"], -1).T
        x = x + w["visual.patch_embed.proj.bias"]
        position = mx.sum(w["visual.pos_embed.weight"][indices] * interp[..., None], axis=1).astype(x.dtype)
        x = x + position
        inv = c.get("rope_parameters", {}).get("rope_theta", 10000.0) ** (-mx.arange(0, width // 2, 2, dtype=mx.float32) / (width // 2))
        angle = coords.astype(mx.float32)[..., None] * inv
        angle = angle.reshape(-1, width // 2)
        cos, sin = mx.cos(angle)[:, None], mx.sin(angle)[:, None]

        def rotate(v):
            dtype = v.dtype
            a, b = mx.split(v.astype(mx.float32), 2, axis=-1)
            return mx.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(dtype)

        for i in range(c["depth"]):
            prefix = f"visual.blocks.{i}"
            z = self.norm(x, prefix + ".norm1")
            qkv = self.model.linear(z, prefix + ".attn.qkv").reshape(-1, 3, c["num_heads"], width)
            q, k, v = (qkv[:, j] for j in range(3))
            q, k = rotate(q), rotate(k)
            parts, offset = [], 0
            for length in lengths:
                args = [a[offset:offset + length].transpose(1, 0, 2)[None] for a in (q, k, v)]
                part = mx.fast.scaled_dot_product_attention(*args, scale=width ** -0.5)
                parts.append(part[0].transpose(1, 0, 2).reshape(length, -1))
                offset += length
            x = x + self.model.linear(mx.concatenate(parts, axis=0), prefix + ".attn.proj")
            z = self.norm(x, prefix + ".norm2")
            z = self.model.linear(z, prefix + ".mlp.linear_fc1")
            x = x + self.model.linear(gelu(z, approximate=True), prefix + ".mlp.linear_fc2")
            mx.eval(x)
        x = self.norm(x, "visual.merger.norm").reshape(-1, c["hidden_size"] * c["spatial_merge_size"] ** 2)
        x = self.model.linear(gelu(self.model.linear(x, "visual.merger.linear_fc1")), "visual.merger.linear_fc2")
        mx.eval(x)
        return x

    def inputs(self, encoded):
        features = self(encoded.pixels, encoded.grid)
        ids = encoded.ids
        image_token = self.model.config["image_token_id"]
        token_positions = np.flatnonzero(ids == image_token)
        if len(token_positions) != features.shape[0]:
            raise ValueError("Image positions and visual features differ")
        embeddings = self.model.weights["language_model.embed_tokens.weight"][mx.array(ids)]
        embeddings[mx.array(token_positions)] = features
        positions = np.zeros((3, len(ids)), dtype=np.int32)
        offset, current, image = 0, 0, 0
        merge = self.config["spatial_merge_size"]
        while offset < len(ids):
            if ids[offset] != image_token:
                end = offset + 1
                while end < len(ids) and ids[end] != image_token:
                    end += 1
                positions[:, offset:end] = np.arange(end - offset)[None] + current
                current += end - offset
            else:
                temporal, height, width = map(int, encoded.grid[image])
                h, w = height // merge, width // merge
                end = offset + temporal * h * w
                coords = np.stack(np.meshgrid(np.arange(temporal), np.arange(h), np.arange(w), indexing="ij")).reshape(3, -1)
                positions[:, offset:end] = coords + current
                current += max(h, w)
                image += 1
            offset = end
        if len(encoded.keys) > 1:
            context = encoded.segments[0]
            offset = context
            start = positions[:, context - 1:context] + 1
            for length in encoded.segments[1:]:
                positions[:, offset:offset + length] = start + np.arange(length)[None]
                offset += length
        return embeddings[None], mx.array(positions)

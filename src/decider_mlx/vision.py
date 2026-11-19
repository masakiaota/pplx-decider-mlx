"""Qwen3.5 image encoder and the multi-axis (M-RoPE) positions of image requests."""
import math

import mlx.core as mx
import numpy as np

BLOCK_WEIGHTS = ('norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias', 'attn.proj.weight', 'attn.proj.bias',
                 'norm2.weight', 'norm2.bias', 'mlp.linear_fc1.weight', 'mlp.linear_fc1.bias', 'mlp.linear_fc2.weight',
                 'mlp.linear_fc2.bias')
MAX_BLOCK_GRAPHS = 64


def gelu(x, approximate=False):
    if approximate:
        return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)))
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


def layout(grid, side, merge):
    """Patch coordinates, bilinear position-embedding gather indices and weights, and patches per image."""
    coords, gather, weights, lengths = [], [], [], []
    for temporal, height, width in np.asarray(grid).tolist():
        bh, bw, dh, dw = np.meshgrid(np.arange(height // merge), np.arange(width // merge), np.arange(merge), np.arange(merge), indexing='ij')
        h = (bh * merge + dh).reshape(-1); w = (bw * merge + dw).reshape(-1)
        rh = h * (side - 1) / max(height - 1, 1); rw = w * (side - 1) / max(width - 1, 1)
        h0 = np.floor(rh).astype(np.int64); w0 = np.floor(rw).astype(np.int64)
        g, wt = [], []
        for a in (0, 1):
            for b in (0, 1):
                g.append(np.minimum(h0 + a, side - 1) * side + np.minimum(w0 + b, side - 1))
                wt.append((1 - np.abs(rh - h0 - a)) * (1 - np.abs(rw - w0 - b)))
        c = np.stack([h, w], 1); g = np.stack(g, 1); wt = np.stack(wt, 1)
        for _ in range(temporal):
            lengths.append(height * width); coords.append(c); gather.append(g); weights.append(wt)
    return (mx.array(np.concatenate(coords).astype(np.int32)), mx.array(np.concatenate(gather).astype(np.int32)),
            mx.array(np.concatenate(weights).astype(np.float32)), lengths)


def _attention(q, k, v, lengths, width):
    parts, offset = [], 0
    for length in lengths:
        args = [a[offset:offset + length].transpose(1, 0, 2)[None] for a in (q, k, v)]
        part = mx.fast.scaled_dot_product_attention(*args, scale=width ** -0.5)
        parts.append(part[0].transpose(1, 0, 2).reshape(length, -1))
        offset += length
    return mx.concatenate(parts, axis=0)


def _rotate(v, cos, sin):
    dtype = v.dtype
    a, b = mx.split(v.astype(mx.float32), 2, axis=-1)
    return mx.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(dtype)


def _block_graph(lengths, heads, width):
    def block(x, cos, sin, n1w, n1b, qw, qb, pw, pb, n2w, n2b, f1w, f1b, f2w, f2b):
        z = mx.fast.layer_norm(x, n1w, n1b, eps=1e-6)
        qkv = (z @ qw.T + qb).reshape(-1, 3, heads, width)
        q, k, v = (qkv[:, j] for j in range(3))
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        x = x + (_attention(q, k, v, lengths, width) @ pw.T + pb)
        z = mx.fast.layer_norm(x, n2w, n2b, eps=1e-6)
        z = z @ f1w.T + f1b
        return x + (gelu(z, approximate=True) @ f2w.T + f2b)
    return mx.compile(block)


class Vision:
    def __init__(self, backbone, compiled):
        self.model = backbone
        self.config = backbone.config['vision_config']
        self.compiled = compiled      # the vision blocks as one compiled function (Flash); eager blocks otherwise
        self.graphs = {}

    def norm(self, x, name):
        w = self.model.weights
        return mx.fast.layer_norm(x, w[name + '.weight'], w[name + '.bias'], eps=1e-6)

    def __call__(self, pixels, grid):
        self.model.load_vision()
        c = self.config
        w = self.model.weights
        width = c['hidden_size'] // c['num_heads']
        coords, indices, interp, lengths = layout(grid, int(math.sqrt(c['num_position_embeddings'])), c['spatial_merge_size'])
        x = mx.array(pixels).astype(self.model.dtype) @ w['visual.patch_embed.proj.weight'].reshape(c['hidden_size'], -1).T
        x = x + w['visual.patch_embed.proj.bias']
        position = mx.sum(w['visual.pos_embed.weight'][indices] * interp[..., None], axis=1).astype(x.dtype)
        x = x + position
        inv = c.get('rope_parameters', {}).get('rope_theta', 10000.0) ** (-mx.arange(0, width // 2, 2, dtype=mx.float32) / (width // 2))
        angle = coords.astype(mx.float32)[..., None] * inv
        angle = angle.reshape(-1, width // 2)
        cos, sin = mx.cos(angle)[:, None], mx.sin(angle)[:, None]
        interval = self.model.sync_interval(lengths)
        if self.compiled:
            key = tuple(lengths)
            graph = self.graphs.get(key)
            if graph is None:
                if len(self.graphs) >= MAX_BLOCK_GRAPHS:
                    del self.graphs[next(iter(self.graphs))]
                graph = self.graphs[key] = _block_graph(key, c['num_heads'], width)
        for i in range(c['depth']):
            prefix = f'visual.blocks.{i}'
            if self.compiled:
                x = graph(x, cos, sin, *[w[f'{prefix}.{n}'] for n in BLOCK_WEIGHTS])
            else:
                z = self.norm(x, prefix + '.norm1')
                qkv = self.model.linear(z, prefix + '.attn.qkv').reshape(-1, 3, c['num_heads'], width)
                q, k, v = (qkv[:, j] for j in range(3))
                q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
                x = x + self.model.linear(_attention(q, k, v, lengths, width), prefix + '.attn.proj')
                z = self.norm(x, prefix + '.norm2')
                z = self.model.linear(z, prefix + '.mlp.linear_fc1')
                x = x + self.model.linear(gelu(z, approximate=True), prefix + '.mlp.linear_fc2')
            if interval and (i + 1) % interval == 0:
                mx.eval(x)
        x = self.norm(x, 'visual.merger.norm').reshape(-1, c['hidden_size'] * c['spatial_merge_size'] ** 2)
        x = self.model.linear(gelu(self.model.linear(x, 'visual.merger.linear_fc1')), 'visual.merger.linear_fc2')
        mx.eval(x)
        return x

    def inputs(self, encoded, features):
        """(1, T, D) embeddings with the image features in place, and (3, T) M-RoPE positions."""
        ids = encoded.ids
        image_token = self.model.config['image_token_id']
        token_positions = np.flatnonzero(ids == image_token)
        if len(token_positions) != features.shape[0]:
            raise ValueError('Image positions and visual features differ')
        embeddings = self.model.weights['language_model.embed_tokens.weight'][mx.array(ids)]
        embeddings[mx.array(token_positions)] = features
        positions = np.zeros((3, len(ids)), dtype=np.int32)
        offset, current, image = 0, 0, 0
        merge = self.config['spatial_merge_size']
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
                coords = np.stack(np.meshgrid(np.arange(temporal), np.arange(h), np.arange(w), indexing='ij')).reshape(3, -1)
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

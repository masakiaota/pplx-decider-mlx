"""The Qwen3.5 dense backbone of Decider v1.3 on packed sequences, and the per-model linear arithmetic.

All valid tokens of a forward are packed into one row (no padding). `Layout` describes the packing: each sequence is
a context followed by questions; a question attends to its context and to itself, and its DeltaNet state is forked
from the end of the context. The last full-attention layer only computes the readout marker rows.
"""
from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from .kernels import attention as attention_kernels
from .kernels import deltanet

METAL_ATTENTION_LIMIT = 512   # chunks up to this length use the segment attention kernel; longer ones the tiled kernel


# ---------------------------------------------------------------- weights
_DTYPES = {'BF16': (mx.bfloat16, 2), 'F16': (mx.float16, 2), 'F32': (mx.float32, 4),
           'U8': (mx.uint8, 1), 'U32': (mx.uint32, 4), 'I32': (mx.int32, 4), 'I64': (mx.int64, 8)}


class MappedSafetensors:
    """Read-only, zero-copy safetensors: the file is memory-mapped and each tensor is a view of it.
    The files must stay unchanged while the arrays exist (checked on every load of the same path)."""

    def __init__(self):
        self.files = {}

    def __call__(self, path):
        path = Path(path).resolve()
        if path not in self.files:
            stat = path.stat()
            with path.open('rb') as source:
                header_size = struct.unpack('<Q', source.read(8))[0]
                if not 2 <= header_size <= min(stat.st_size - 8, 32_000_000):
                    raise ValueError(f'Invalid or unexpectedly large safetensors header: {path}')
                header = json.loads(source.read(header_size))
            base = 8 + header_size
            specs = {}
            for name, spec in header.items():
                if name == '__metadata__':
                    continue
                dtype, width = _DTYPES[spec['dtype']]
                begin, end = spec['data_offsets']; shape = tuple(spec['shape'])
                if not (0 <= begin <= end <= stat.st_size - base) or any(s < 0 for s in shape):
                    raise ValueError(f'Invalid tensor bounds in {path}')
                if end - begin != math.prod(shape) * width or (base + begin) % width or (base + end) % width:
                    raise ValueError(f'Tensor payload not aligned to its dtype: {name}')
                specs[name] = (base + begin, base + end, dtype, shape)
            # The widest zero-copy container compatible with every tensor boundary (a U8 dimension over 2 GiB
            # exceeds MLX's int32 shape limit); views keep each tensor's own dtype.
            boundaries = [stat.st_size, *[offset for spec in specs.values() for offset in spec[:2]]]
            unit = next(size for size in (8, 4, 2, 1) if all(offset % size == 0 for offset in boundaries))
            if stat.st_size // unit > 2 ** 31 - 1:
                raise ValueError('No aligned zero-copy container fits the MLX int32 shape limit')
            owner = np.memmap(path, mode='r', dtype={8: np.uint64, 4: np.uint32, 2: np.uint16, 1: np.uint8}[unit])
            raw = mx.asarray(owner, copy=False)
            values = {name: raw[begin // unit:end // unit].view(dtype).reshape(shape) for name, (begin, end, dtype, shape) in specs.items()}
            self.files[path] = (owner, values, (stat.st_ino, stat.st_size, stat.st_mtime_ns))
        owner, values, fingerprint = self.files[path]
        stat = path.stat()
        if fingerprint != (stat.st_ino, stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f'A mapped model file changed while in use: {path}')
        return dict(values)

    @property
    def mapped_bytes(self):
        return sum(owner.nbytes for owner, _, _ in self.files.values())


def shard_files(directory):
    directory = Path(directory)
    index = directory / 'model.safetensors.index.json'
    files = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else sorted(p.name for p in directory.glob('model*.safetensors'))
    if not files:
        raise FileNotFoundError(f'No model weight shards in {directory}')
    return files


# ---------------------------------------------------------------- backbone
def rms(x, weight, eps=1e-6, zero_centered=True):
    xf = x.astype(mx.float32)
    w = weight.astype(mx.float32) + (1.0 if zero_centered else 0.0)
    return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps) * w).astype(x.dtype)


def silu(x):
    return x * mx.sigmoid(x)


class Backbone:
    def __init__(self, directory, loader, *, delta_threadgroup):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / 'config.json').read_text())
        self.text = self.config['text_config']
        if self.config['model_type'] != 'qwen3_5':
            raise ValueError('Only the two dense Decider v1.3 backbones are supported')
        self.dtype = mx.bfloat16
        self.loader = loader
        self.delta_threadgroup = tuple(delta_threadgroup)
        self.locate_enabled = False      # Locate forwards: Locate weights / LoRA and the marker embedding
        self.locate = None               # LocateWeights once loaded (locate.py)
        metadata = self.directory / 'mlx_config.json'
        self.quantization = json.loads(metadata.read_text()).get('quantization', {}) if metadata.exists() else {}
        self.files = shard_files(self.directory)
        self.weights = {}
        for filename in self.files:
            for name, value in loader(self.directory / filename).items():
                if name.startswith('language_model.'):
                    self.weights[name] = value.astype(self.dtype) if value.dtype in (mx.float16, mx.bfloat16, mx.float32) else value
            mx.eval(self.weights)
        self.readout = loader(self.directory / 'readout.safetensors')['weight'].astype(self.dtype)
        mx.eval(self.readout)
        self.linear = None               # set by the arithmetic (AffineInt8 / W4A8)
        self.last_layer = len(self.text['layer_types']) - 1

    def load_vision(self):
        if 'visual.patch_embed.proj.weight' in self.weights:
            return
        for filename in self.files:
            for name, value in self.loader(self.directory / filename).items():
                if name.startswith('visual.'):
                    self.weights[name] = value.astype(self.dtype)
            mx.eval(self.weights)

    def norm(self, x, name, zero_centered=True):
        return rms(x, self.weights[name + '.weight'], self.text['rms_norm_eps'], zero_centered)

    def rotary(self, x, positions):
        config = self.text['rope_parameters']
        dim = int(self.text['head_dim'] * config.get('partial_rotary_factor', 1.0))
        inv = config['rope_theta'] ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        if positions.ndim == 1:
            angle = positions[:, None].astype(mx.float32) * inv
        else:
            angles = positions[..., None].astype(mx.float32) * inv
            indices = list(range(dim // 2))
            section = config.get('mrope_section', [11, 11, 10])
            dimensions = [1 if i % 3 == 1 and i < section[1] * 3 else 2 if i % 3 == 2 and i < section[2] * 3 else 0 for i in indices]
            angle = mx.stack([angles[d, :, i] for i, d in enumerate(dimensions)], axis=-1)
        cos, sin = mx.cos(angle).astype(x.dtype)[None, None], mx.sin(angle).astype(x.dtype)[None, None]
        first, second = x[..., :dim // 2], x[..., dim // 2:dim]
        return mx.concatenate([first * cos - second * sin, second * cos + first * sin, x[..., dim:]], axis=-1)

    def sync_interval(self, lengths):
        """Eager forwards evaluate after every layer only for large inputs (bounds the lazy graph)."""
        return 1 if (len(lengths) > 8 or max(lengths, default=0) > 5500 or sum(lengths) > 8192) else 0

    # The standard (non-INT8) linear: MLX QMM / BF16 matmul, on the Locate weights for Locate forwards.
    def qmm(self, x, name, weights=None, quantization=None):
        weights = self.weights if weights is None else weights
        quantization = self.quantization if quantization is None else quantization
        weight = weights[name + '.weight']
        if name + '.scales' in weights:
            config = quantization.get(name, quantization)
            return mx.quantized_matmul(x, weight, weights[name + '.scales'], weights.get(name + '.biases'), transpose=True,
                                       group_size=config.get('group_size', 64), bits=config.get('bits', 4), mode=config.get('mode', 'affine'))
        return x @ weight.T


# ---------------------------------------------------------------- arithmetic
_LAYER = re.compile(r'language_model\.layers\.(\d+)\.')


def _layer(name):
    m = _LAYER.match(name)
    return int(m.group(1)) if m else -1


class AffineInt8:
    """Flash: affine8 weights. Matrices with INT8 codes (MLP, in_proj_qkv/z, q_proj) run the INT8 x u8 kernel when the
    call has at least 128 rows; the rest (and Locate, which runs the merged Locate weights) run the standard QMM.
    Sequences holding a score question keep layers in `score_layers` on the standard QMM. When score and other
    sequences share one packed forward, those layers split the rows: score rows standard, the others INT8.

    State (set by the engine around each forward and read at trace time, so it is part of every graph key):
      score: None, or the per-sequence score flags of the forward; mixed: some but not all flags set;
      lengths: the sequence lengths (row runs for the split); bulk: None or the bulk domain ('text'/'score').
    """
    name = 'affine-int8'

    def __init__(self, model, int8_values, score_layers):
        self.model, self.values, self.score_layers = model, int8_values, frozenset(score_layers)
        self.score = None; self.mixed = False; self.lengths = (); self.bulk = None

    def key(self):
        return self.score if self.score is not None else False

    def standard(self, x, name):
        m = self.model
        merged = m.locate_enabled and name in m.locate.modules
        weights = m.locate.weights if merged else m.weights
        result = m.qmm(x, name, weights, m.locate.quantization if merged else None)
        if name + '.bias' in weights:
            result = result + weights[name + '.bias']
        return result

    def _int8(self, x, name, value):
        from .kernels.int8 import linear
        y = linear(x, *value)
        if name + '.bias' in self.model.weights:
            y = y + self.model.weights[name + '.bias']
        return y

    def _runs(self, n):
        runs = []; a = 0
        for length, flag in zip(self.lengths, self.score):
            if runs and runs[-1][2] == flag:
                runs[-1][1] += length
            else:
                runs.append([a, a + length, flag])
            a += length
        if a < n:
            if runs[-1][2] is False:
                runs[-1][1] = n
            else:
                runs.append([a, n, False])   # bucket dummy rows
        return [tuple(r) for r in runs]

    def _split(self, x, name, value):
        # Score rows are gathered into one call, so a score answer does not depend on the non-score questions beside it.
        runs = self._runs(x.shape[0])
        xs = mx.concatenate([x[a:b] for a, b, f in runs if f]); xo = mx.concatenate([x[a:b] for a, b, f in runs if not f])
        ys, yo = self.standard(xs, name), self._int8(xo, name, value)
        parts = []; ps = po = 0
        for a, b, f in runs:
            if f:
                parts.append(ys[ps:ps + b - a]); ps += b - a
            else:
                parts.append(yo[po:po + b - a]); po += b - a
        return mx.concatenate(parts)

    def __call__(self, x, name):
        value = self.values.get(name)
        if self.model.locate_enabled or value is None or math.prod(x.shape[:-1]) < 128:
            return self.standard(x, name)
        layer = _layer(name)
        if self.bulk is not None:
            # fixed-budget packs: the pruned last layer has a fixed multiple of 128 rows; its q_proj stays standard
            # as in the per-request route, where it has fewer than 128 rows
            if (self.bulk == 'score' and layer in self.score_layers) or (layer == self.model.last_layer and name.endswith('.q_proj')):
                return self.standard(x, name)
            return self._int8(x, name, value)
        if self.score is not None and layer in self.score_layers:
            if self.mixed and x.ndim == 2:
                return self._split(x, name, value)
            return self.standard(x, name)
        return self._int8(x, name, value)


class W4A8:
    """27B: the row-Q4 tensors (MLP, in_proj_qkv/z, out_proj, q_proj, o_proj) on the W4A8 s4 kernel for any row count;
    k/v group Q4 on the standard QMM, in_proj_a/b BF16. Locate adds the BF16 LoRA."""
    name = 'w4a8-s4'

    def __init__(self, model, rows):
        self.model, self.rows = model, rows

    def key(self):
        return False

    def __call__(self, x, name):
        from .kernels.w4a8_s4 import linear
        m = self.model; w = m.weights
        if name in self.rows:
            y = linear(x, *self.rows[name])
        else:
            y = m.qmm(x, name)
        if name + '.bias' in w:
            y = y + w[name + '.bias']
        if m.locate_enabled and name + '.lora_a' in m.locate.lora:
            lora = m.locate.lora
            h = x.astype(lora[name + '.lora_a'].dtype) @ lora[name + '.lora_a'].T
            y = y + ((h @ lora[name + '.lora_b'].T) * m.locate.lora_scale).astype(y.dtype)
        return y


# ---------------------------------------------------------------- packed forward
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
            if len(chunk.keys) == 1:
                self.groups.append((start, end, start, end, 0, 0))
            else:
                self.groups.append((start, context, start, end, 0, 0))
            for size in chunk.segments[1:]:
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


# Layout tensors passed as graph inputs (bucket graphs), in this order.
LAYOUT_TENSORS = ('starts', 'owners', 'contexts', 'question_offsets', 'question_starts', 'question_ends', 'markers', 'ranges',
                  'blocks', 'prefixes', 'positions')


def delta(model, x, name, layout):
    c = model.text
    kh, vh = c['linear_num_key_heads'], c['linear_num_value_heads']
    kd, vd = c['linear_key_head_dim'], c['linear_value_head_dim']
    projected = model.linear(x, name + '.in_proj_qkv')
    beta = mx.sigmoid(model.linear(x, name + '.in_proj_b'))
    a = model.linear(x, name + '.in_proj_a').astype(mx.float32) + model.weights[name + '.dt_bias'].astype(mx.float32)
    g = -mx.exp(model.weights[name + '.A_log'].astype(mx.float32)) * mx.logaddexp(a, mx.array(0.0))
    convolved = deltanet.convolution(projected, model.weights[name + '.conv1d.weight'][:, 0],
                                     layout.starts, layout.owners, layout.blocks, layout.prefixes)
    q, k, v = mx.split(convolved, [kh * kd, 2 * kh * kd], axis=-1)
    q, k = q.reshape(-1, kh, kd).astype(mx.float32), k.reshape(-1, kh, kd).astype(mx.float32)
    v = v.reshape(-1, vh, vd).astype(mx.float32)
    q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6) / math.sqrt(kd)
    k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    out = deltanet.delta(q, k, v, mx.exp(g), beta.astype(mx.float32), layout.starts, layout.contexts,
                         layout.question_offsets, layout.question_starts, layout.question_ends,
                         threadgroup=model.delta_threadgroup).astype(x.dtype)
    z = model.linear(x, name + '.in_proj_z').reshape(-1, vh, vd)
    xf = out.astype(mx.float32)
    normalized = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + c['rms_norm_eps'])).astype(x.dtype)
    out = (normalized * model.weights[name + '.norm.weight'] * silu(z.astype(mx.float32))).astype(x.dtype)
    return model.linear(out.reshape(x.shape[0], -1), name + '.out_proj')


def attention(model, x, name, layout, last):
    c = model.text
    heads, kvheads, width = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
    source = x[layout.markers] if last else x
    q, gate = mx.split(model.linear(source, name + '.q_proj').reshape(-1, heads, 2 * width), 2, axis=-1)
    q = model.norm(q, name + '.q_norm')
    k = model.norm(model.linear(x, name + '.k_proj').reshape(-1, kvheads, width), name + '.k_norm')
    v = model.linear(x, name + '.v_proj').reshape(-1, kvheads, width)
    q_positions = layout.positions[:, layout.markers] if last else layout.positions
    q = model.rotary(q[None].transpose(0, 2, 1, 3), q_positions).transpose(0, 2, 1, 3)[0]
    k = model.rotary(k[None].transpose(0, 2, 1, 3), layout.positions).transpose(0, 2, 1, 3)[0]
    if not last and max(layout.lengths) > METAL_ATTENTION_LIMIT:
        key_parts, value_parts, q_lengths, k_lengths = [], [], [], []
        for qa, qb, ka, kb, kc, kd in layout.groups:
            key_parts.append(k[ka:kb] if kc == kd else mx.concatenate([k[ka:kb], k[kc:kd]]))
            value_parts.append(v[ka:kb] if kc == kd else mx.concatenate([v[ka:kb], v[kc:kd]]))
            q_lengths.append(qb - qa)
            k_lengths.append(kb - ka + kd - kc)
        out = attention_kernels.tiled_attention(q, mx.concatenate(key_parts), mx.concatenate(value_parts), q_lengths, k_lengths)
    elif max(layout.lengths) <= METAL_ATTENTION_LIMIT:
        ranges = layout.ranges[layout.markers] if last else layout.ranges
        out = attention_kernels.segment_attention(q, k, v, ranges)
    else:
        parts = []
        for qa, qb, ka, kb, kc, kd in [(i, i + 1, *group) for i, group in enumerate(layout.marker_groups)]:
            query = q[qa:qb][None].transpose(0, 2, 1, 3)
            keys = k[ka:kb] if kc == kd else mx.concatenate([k[ka:kb], k[kc:kd]])
            values = v[ka:kb] if kc == kd else mx.concatenate([v[ka:kb], v[kc:kd]])
            out = mx.fast.scaled_dot_product_attention(query, keys[None].transpose(0, 2, 1, 3),
                                                       values[None].transpose(0, 2, 1, 3), scale=width ** -0.5)
            parts.append(out.transpose(0, 2, 1, 3)[0])
        out = mx.concatenate(parts)
    out = out.reshape(source.shape[0], -1) * mx.sigmoid(gate.reshape(source.shape[0], -1))
    return model.linear(out, name + '.o_proj')


def locate_marker(model, x, ids):
    """Locate forwards: the <|locate|> marker token takes its trained embedding."""
    if model.locate_enabled:
        x = mx.where((ids == model.locate.marker)[:, None], model.locate.marker_embedding.astype(x.dtype), x)
    return x


def layers(model, x, layout, return_all, interval=0):
    """Every decoder layer and the final norm. Without return_all the last full-attention layer keeps only the
    marker rows. Returns (hidden, pruned). interval > 0 evaluates every `interval` layers (eager forwards only)."""
    pruned = False
    for index, kind in enumerate(model.text['layer_types']):
        prefix = f'language_model.layers.{index}'
        normalized = model.norm(x, prefix + '.input_layernorm')
        last = not return_all and index == model.last_layer
        if kind == 'full_attention':
            update = attention(model, normalized, prefix + '.self_attn', layout, last)
            if last:
                x = x[layout.markers]
                pruned = True
        else:
            update = delta(model, normalized, prefix + '.linear_attn', layout)
        x = x + update
        normalized = model.norm(x, prefix + '.post_attention_layernorm')
        x = x + model.linear(silu(model.linear(normalized, prefix + '.mlp.gate_proj')) * model.linear(normalized, prefix + '.mlp.up_proj'), prefix + '.mlp.down_proj')
        if interval and (index + 1) % interval == 0:
            mx.eval(x)
    return model.norm(x, 'language_model.norm'), pruned


def logits(model, x, markers, pruned):
    return ((x if pruned else x[markers]) @ model.readout.T).astype(mx.float32)


def per_chunk(values, chunks):
    """(questions, R) marker rows -> (chunks, max questions, R), zero padded."""
    maximum, result, offset = max(len(c.keys) for c in chunks), [], 0
    for chunk in chunks:
        count = len(chunk.keys)
        result.append(mx.pad(values[offset:offset + count], [(0, maximum - count), (0, 0)]))
        offset += count
    return mx.stack(result)


def stacked_hidden(x, offsets, count):
    """Flat hidden states -> (count, longest, D), zero padded, for the first `count` sequences."""
    pairs = list(zip(offsets[:count], offsets[1:count + 1]))
    maximum = max(b - a for a, b in pairs)
    return mx.stack([mx.pad(x[a:b], [(0, maximum - (b - a)), (0, 0)]) for a, b in pairs])


def forward_core(model, layout, ids, multimodal=None, return_all=False, compiled=False):
    """A packed forward: (chunks, questions, R) readout logits, or (chunks, longest, D) hidden states with return_all."""
    chunks = layout.chunks
    table = model.weights['language_model.embed_tokens.weight']
    if multimodal is None:
        x = table[ids]
    else:
        x = mx.concatenate([table[mx.array(chunk.ids)] if value is None else value[0][0] for chunk, value in zip(chunks, multimodal)])
    x = locate_marker(model, x, ids)
    # Whole-graph compilation forbids evaluation inside the transformation.
    x, pruned = layers(model, x, layout, return_all, 0 if compiled else model.sync_interval(layout.lengths))
    if return_all:
        return stacked_hidden(x, layout.offsets, len(layout.lengths))
    return per_chunk(logits(model, x, layout.markers, pruned), chunks)

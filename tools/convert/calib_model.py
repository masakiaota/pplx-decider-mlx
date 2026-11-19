"""Step 4b (27B): the calibration model, a 4-bit model whose activations gram.py collects.

    python tools/convert/calib_model.py ORIGINAL Q4 STAGE1 OUT

The row-wise 4-bit weights are compensated with H = E[x x^T] of each layer's input. The published H was collected
by running a 4-bit model, not the BF16 original (which needs about 57 GB): Q4 (quantize.py --bits 4) with the MLP
gate/up projections of layers 0-62 improved by this step. Reproducing the published weights needs this exact model.

1. Factors: Q4 runs the requests of STAGE1 (calib_set.py stage1.jsonl) in the calibration engine (calib_engine/,
   the research engine these weights were made with), separate mode, temperature 1. For every gate_proj input of
   layers 0-62, H is accumulated per block of 64 input channels (FP32), divided by the token count, damped by 1%
   of its mean diagonal, inverted and factorized (upper Cholesky factor of H^-1, float64) -> OUT/factors.safetensors.
2. Compensation: for gate_proj and up_proj of layers 0-62, the original BF16 weight is re-rounded within each
   64-column group in natural order (GPTQ inside the group, no propagation across groups), keeping the scales and
   biases of Q4, so only the 4-bit codes change. OUT gets copies (APFS clones) of the Q4 shards with these codes,
   and the other files of Q4.

About 5 minutes; MLX peak about 25 GB (the engine maps Q4, 17 GB).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

os.environ['MLX_ENABLE_TF32'] = '0'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from st import clone, header  # noqa: E402

OPTIONS = dict(delta_backend='native', share_context=True, prune_last=True, packed=True, pack_single=True,
               sort_by_length=True, batch_size=8, batch_tokens=8192, eval_interval=0, norm_mode='plain',
               image_cache=False, memory_limit_gb=32)


def factors(q4, stage1, out):
    from calib_engine import Engine
    from calib_engine.mapped_safetensors import MappedSafetensors
    mx.set_memory_limit(32_000_000_000)
    mx.set_cache_limit(256_000_000)
    loader = MappedSafetensors(32_000_000_000)
    mx.load = loader
    engine = Engine(q4, **OPTIONS)
    original = engine.backbone.linear
    grams, counts = {}, {}

    def linear(backbone, x, name):
        if name.endswith('.mlp.gate_proj') and int(name.split('.')[2]) < 63:
            layer = int(name.split('.')[2])
            flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            t, k = flat.shape
            z = flat.reshape(t, k // 64, 64).transpose(1, 2, 0)
            h = z @ z.swapaxes(-1, -2)
            grams[layer] = grams[layer] + h if layer in grams else h
            counts[layer] = counts.get(layer, 0) + t
            mx.eval(grams[layer])
        return original(x, name)
    engine.backbone.linear = types.MethodType(linear, engine.backbone)
    for row in stage1:
        engine.decide(row['request'], temperature=1., execution_mode='separate')
        print(json.dumps({'calibrated': row['id']}), flush=True)
    assert sorted(grams) == list(range(63))
    result = {}
    for layer in sorted(grams):
        host = np.asarray(grams[layer]).astype(np.float64) / counts[layer]
        damp = .01 * np.mean(np.diagonal(host, axis1=-2, axis2=-1), axis=-1)
        host += np.maximum(damp, 1e-12)[:, None, None] * np.eye(64)[None, :, :]
        inverse = np.linalg.inv(host)
        u = np.swapaxes(np.linalg.cholesky(inverse), -1, -2).copy()
        if not np.isfinite(u).all():
            raise ValueError(f'non-finite factor in layer {layer}')
        result[str(layer)] = mx.array(u.astype(np.float32))
    mx.eval(result)
    mx.save_safetensors(str(out / 'factors.safetensors'), result)
    print(json.dumps({'factors': len(result), 'tokens': counts[0]}), flush=True)
    mx.load = loader.original_load
    del engine, grams
    mx.clear_cache()


def compensate(w, baseline, u):
    """Natural-order GPTQ within each 64-column group of affine 4-bit weights; scales and biases unchanged."""
    n, k = w.shape
    g = k // 64
    scales = baseline[1].astype(mx.float32)
    biases = baseline[2].astype(mx.float32)
    invscale = mx.where(scales != 0, 1 / scales, 0)
    work = w.astype(mx.float32).reshape(n, g, 64)
    packed, word = [], mx.zeros((n, g), mx.uint32)
    for col in range(64):
        value = work[:, :, col]
        code = mx.clip(mx.round((value - biases) * invscale), 0, 15).astype(mx.uint32)
        quantized = (code.astype(mx.float32) * scales + biases).astype(mx.bfloat16).astype(mx.float32)
        error = (value - quantized) / u[:, col, col][None, :]
        tail = work[:, :, col:] - error[:, :, None] * u[None, :, col, col:]
        work = mx.concatenate([work[:, :, :col], tail], axis=-1) if col else tail
        word = word | (code << (4 * (col % 8)))
        mx.eval(work, word)
        if col % 8 == 7:
            packed.append(word)
            word = mx.zeros((n, g), mx.uint32)
    codes = mx.stack(packed, axis=-1).reshape(n, k // 8)
    mx.eval(codes)
    return codes


def view(path, name, hdr=None):
    """One tensor of a safetensors file as an MLX array over a read-only memmap (BF16 viewed as bfloat16)."""
    base, h = hdr or header(path)
    spec = h[name]
    dtype = {'BF16': np.uint16, 'U32': np.uint32, 'U8': np.uint8, 'F32': np.float32}[spec['dtype']]
    owner = np.memmap(path, mode='r', dtype=dtype, offset=base + spec['data_offsets'][0], shape=tuple(spec['shape']))
    value = mx.array(np.array(owner)) if (base + spec['data_offsets'][0]) % np.dtype(dtype).itemsize else mx.asarray(owner, copy=False)
    return (value.view(mx.bfloat16) if spec['dtype'] == 'BF16' else value), owner


def convert(original, q4, out):
    mx.set_memory_limit(12_000_000_000)
    mx.set_cache_limit(256_000_000)
    index = json.loads((q4 / 'model.safetensors.index.json').read_text())
    oindex = json.loads((original / 'model.safetensors.index.json').read_text())['weight_map']
    shards = sorted(set(index['weight_map'].values()))
    fac = mx.load(str(out / 'factors.safetensors'))
    for path in q4.iterdir():
        if path.name not in shards and path.is_file():
            clone(path, out / path.name)
    for path in sorted((q4 / 'locate-span').iterdir()):
        clone(path, out / 'locate-span' / path.name)
    count = 0
    for shard in shards:
        src, dst = q4 / shard, out / shard
        clone(src, dst)
        hdr = header(src)
        base, h = hdr
        names = [n for n in h if n.startswith('language_model.layers.') and n.endswith(('.mlp.gate_proj.weight', '.mlp.up_proj.weight'))
                 and int(n.split('.')[2]) < 63]
        with dst.open('r+b') as stream:
            for name in names:
                layer = int(name.split('.')[2])
                prefix = name.removesuffix('.weight')
                w, owner = view(original / oindex[name], name)
                pieces = [view(src, prefix + '.' + s, hdr) for s in ('weight', 'scales', 'biases')]
                codes = np.asarray(compensate(w, tuple(p[0] for p in pieces), fac[str(layer)]))
                assert codes.nbytes == h[name]['data_offsets'][1] - h[name]['data_offsets'][0]
                stream.seek(base + h[name]['data_offsets'][0])
                codes.tofile(stream)
                count += 1
                del w, owner, pieces, codes
                mx.clear_cache()
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({'shard': shard, 'compensated': len(names)}), flush=True)
    assert count == 126, count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original', type=Path)
    ap.add_argument('q4', type=Path)
    ap.add_argument('stage1', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--factors-only', action='store_true', help='stop after OUT/factors.safetensors')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stage1 = [json.loads(line) for line in a.stage1.read_text().splitlines() if line.strip()]
    if not (a.out / 'factors.safetensors').exists():
        factors(a.q4, stage1, a.out)
    if not a.factors_only:
        convert(a.original, a.q4, a.out)
    print(json.dumps({'out': str(a.out)}))


if __name__ == '__main__':
    main()

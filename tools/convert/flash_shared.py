"""Step 2 (Flash): affine 8-bit -> "shared affine8" main weights and row sums.

    python tools/convert/flash_shared.py Q8 OUT

Q8 is the output of `quantize.py --bits 8`. For the MLP (gate, up, down) and input projections (DeltaNet
in_proj_qkv and in_proj_z, attention q_proj) of layers 0-30, the 8-bit weights are re-quantized per output row to
symmetric INT8, from their dequantized FP32 values:

    scale = max(max|w_row| / 127, 1e-30)        q = clip(round(w / scale), -127, 127)

and stored in place, in the same tensors and shapes: codes q + 128 as unsigned 8-bit, the row scale rounded to
BF16 and repeated for every group of 64, bias = -128 * scale. MLX's quantized matmul (group-64 affine 8-bit) and
the engine's INT8 x INT8 matmul then read the same bytes. Layer 31 and the other tensors stay as in Q8.
row-sums.safetensors holds sum(q) per row (INT32) for the INT8 matmul epilogue.

The shards are copied (an APFS clone on macOS) and only the converted byte ranges are rewritten. About 20 s.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('MLX_ENABLE_TF32', '0')
import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st import clone, header  # noqa: E402

INPUT = ('.linear_attn.in_proj_qkv.weight', '.linear_attn.in_proj_z.weight', '.self_attn.q_proj.weight')


def selected(name):
    if not name.startswith('language_model.layers.') or not name.endswith('.weight'):
        return False
    return int(name.split('.')[2]) < 31 and ('.mlp.' in name or name.endswith(INPUT))


def row_int8(path, base_header, name):
    """Row-wise symmetric INT8 of the dequantized affine 8-bit tensor `name` (codes int8 [N, K], scales F32 [N])."""
    base, h = base_header
    parts = []
    for suffix in ('weight', 'scales', 'biases'):
        spec = h[f'{name}.{suffix}']
        b, e = spec['data_offsets']
        raw = np.fromfile(path, dtype=np.uint8, count=e - b, offset=base + b)
        value = mx.array(raw).view(mx.uint32 if suffix == 'weight' else mx.bfloat16).reshape(spec['shape'])
        parts.append(value)
    dense = mx.dequantize(*parts, group_size=64, bits=8, mode='affine').astype(mx.float32)
    scale = mx.maximum(mx.max(mx.abs(dense), axis=-1) / 127, 1e-30)
    codes = mx.clip(mx.round(dense / scale[:, None]), -127, 127).astype(mx.int8)
    mx.eval(codes, scale)
    return np.asarray(codes), np.asarray(scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('q8', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f'{a.out}: must be empty')
    a.out.mkdir(parents=True, exist_ok=True)
    mx.set_memory_limit(4_000_000_000)
    mx.set_cache_limit(100_000_000)
    sums, count = {}, 0
    for path in sorted(a.q8.glob('model*.safetensors')):
        base_header = header(path)
        base, h = base_header
        names = [k[:-7] for k in h if selected(k)]
        target = a.out / path.name
        clone(path, target)
        with target.open('r+b') as out:
            for name in names:
                q, s = row_int8(path, base_header, name)
                n, k = q.shape
                codes = (q.astype(np.int16) + 128).astype(np.uint8)
                rows = mx.array(s).astype(mx.bfloat16)
                sc = mx.repeat(rows[:, None], k // 64, axis=1)
                bi = -128 * sc
                mx.eval(sc, bi)
                data = {'weight': codes.tobytes(), 'scales': np.asarray(sc.view(mx.uint16)).tobytes(),
                        'biases': np.asarray(bi.view(mx.uint16)).tobytes()}
                for suffix, raw in data.items():
                    b, e = h[f'{name}.{suffix}']['data_offsets']
                    assert e - b == len(raw), name
                    out.seek(base + b)
                    out.write(raw)
                sums[name + '.row_sum'] = mx.array(np.sum(q, axis=1, dtype=np.int32))
                mx.eval(sums[name + '.row_sum'])
                count += 1
                mx.clear_cache()
            out.flush()
            os.fsync(out.fileno())
        print(json.dumps({'shard': path.name, 'converted': len(names)}), flush=True)
    assert count == 148, count
    mx.save_safetensors(str(a.out / 'row-sums.safetensors'), sums)
    for name in ('model.safetensors.index.json', 'mlx_config.json'):
        shutil.copyfile(a.q8 / name, a.out / name)
    print(json.dumps({'out': str(a.out), 'matrices': count}))


if __name__ == '__main__':
    main()

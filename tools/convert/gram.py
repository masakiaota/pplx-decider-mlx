"""Step 5 (27B): H = E[x x^T] of the input of every row-wise 4-bit layer, over the calibration mix.

    python tools/convert/gram.py CALIB_MODEL MIX OUT [--passes 0-7]

CALIB_MODEL is the output of calib_model.py, MIX the mix.jsonl of calib_set.py. The calibration engine
(calib_engine/) answers each request of MIX in order (separate mode, temperature 1, the last layer on all
positions) and, for the layers of the pass, accumulates S = sum_t x_t^T x_t of the inputs of gate/up (mlp_in),
down (down_in), in_proj_qkv/z (lin_in), out_proj (lin_out_in), q_proj (attn_in) and o_proj (attn_o_in) in FP32 on
the GPU, 4,096 rows per update and at every segment boundary of MIX. Labels are not read.

The 64 layers are split into 8 passes of 8 layers (`--passes 0-7`, or e.g. `--passes 3`); pass i writes
OUT/p<i>/L<layer>.<input>.npy (S / T, FP32) and OUT/p<i>/collect.json. One pass takes about 5.5 minutes on an M5
Ultra and needs about 36 GB (the 17 GB calibration model plus 12.6 GB of FP32 sums); the whole H is about 100 GB
of disk. rowq4.py reads the pass folders.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['MLX_ENABLE_TF32'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

OPTIONS = dict(delta_backend='native', share_context=True, prune_last=True, packed=True, pack_single=True,
               sort_by_length=True, batch_size=8, batch_tokens=8192, eval_interval=0, norm_mode='plain',
               image_cache=False, memory_limit_gb=32, max_length=12000)
KEYS = {'mlp.gate_proj': 'mlp_in', 'mlp.down_proj': 'down_in', 'linear_attn.in_proj_qkv': 'lin_in',
        'linear_attn.out_proj': 'lin_out_in', 'self_attn.q_proj': 'attn_in', 'self_attn.o_proj': 'attn_o_in'}
FLUSH = 4096
PASSES = 8


def load(model):
    from calib_engine import Engine
    from calib_engine.mapped_safetensors import MappedSafetensors
    mx.set_memory_limit(32_000_000_000)
    mx.set_cache_limit(1_000_000_000)
    loader = MappedSafetensors(32_000_000_000)
    mx.load = loader
    engine = Engine(model, **OPTIONS)
    engine.loader = loader
    mx.set_memory_limit(48_000_000_000)
    loader.limit_bytes = engine.memory_limit = 48_000_000_000
    engine.backbone.prune_last = False      # the last layer sees every position, not only the decision markers
    return engine


def collect(engine, items, layers, out):
    model = engine.backbone
    original = model.linear
    S, C, buf, nbuf = {}, {}, {}, {}

    def flush(k):
        if not buf.get(k):
            return
        f = mx.concatenate(buf[k]).astype(mx.float32)
        S[k] = f.T @ f if k not in S else mx.addmm(S[k], f.T, f)
        mx.eval(S[k])
        buf[k] = []
        nbuf[k] = 0

    def linear(x, name):
        parts = name.split('.')
        if parts[0] == 'language_model' and parts[1] == 'layers':
            layer, key = int(parts[2]), KEYS.get('.'.join(parts[3:]))
            if key and layer in layers:
                b = mx.contiguous(x.reshape(-1, x.shape[-1]))
                mx.eval(b)
                buf.setdefault((layer, key), []).append(b)
                C[(layer, key)] = C.get((layer, key), 0) + b.shape[0]
                nbuf[(layer, key)] = nbuf.get((layer, key), 0) + b.shape[0]
                if nbuf[(layer, key)] >= FLUSH:
                    flush((layer, key))
        return original(x, name)

    model.linear = linear
    started = time.time()
    try:
        for idx, it in enumerate(items):
            engine.decide(it['request'], temperature=1.0, execution_mode='separate')
            if 'seg' in it and idx + 1 < len(items) and items[idx + 1].get('seg') != it['seg']:
                for k in list(buf):          # segment boundary
                    flush(k)
            if (idx + 1) % 100 == 0:
                print(json.dumps({'items': idx + 1, 'seconds': round(time.time() - started, 1)}), flush=True)
        for k in list(buf):
            flush(k)
    finally:
        model.linear = original
    out.mkdir(parents=True, exist_ok=True)
    for (layer, key), s in S.items():
        np.save(out / f'L{layer}.{key}.npy', np.asarray(s / C[(layer, key)]))
    (out / 'collect.json').write_text(json.dumps(dict(
        layers=sorted(layers), items=len(items), counts={f'L{l}.{k}': c for (l, k), c in C.items()},
        seconds=round(time.time() - started, 1)), indent=1) + '\n')
    S.clear()
    mx.clear_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', type=Path)
    ap.add_argument('mix', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--passes', default=f'0-{PASSES - 1}', help='passes to run: "0-7", "3" or "0,2"')
    a = ap.parse_args()
    spec = a.passes
    passes = list(range(int(spec.split('-')[0]), int(spec.split('-')[1]) + 1)) if '-' in spec else [int(x) for x in spec.split(',')]
    items = [json.loads(line) for line in a.mix.read_text().splitlines() if line.strip()]
    engine = load(a.model)
    per = engine.backbone.text['num_hidden_layers'] // PASSES
    for p in passes:
        target = a.out / f'p{p}'
        if (target / 'collect.json').exists():
            print(json.dumps({'pass': p, 'status': 'done before'}), flush=True)
            continue
        collect(engine, items, set(range(p * per, (p + 1) * per)), target)
        print(json.dumps({'pass': p, 'out': str(target)}), flush=True)


if __name__ == '__main__':
    main()

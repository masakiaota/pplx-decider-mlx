"""Step 6 (27B): row-wise symmetric 4-bit weights with GPTQ-style error compensation, act-order and a row-scale search.

    python tools/convert/rowq4.py ORIGINAL GRAM OUT [--layers 0,3] [--shard I/N]

For each of the 368 target tensors (gate/up/down of every layer; q/o of the attention layers; in_proj_qkv/z and
out_proj of the DeltaNet layers; not k/v) of the original BF16 checkpoint ORIGINAL, with H = E[x x^T] of the
layer's input from GRAM (gram.py; one folder or a comma-separated list, searched in order):

1. Row scale search: for each row pick r in {0.70, 0.72, ..., 1.00} that minimises sum_j H_jj (w_j - s q_j)^2 with
   s = r * max|w_row| / 7 and q = clip(rint(w / s), -7, 7).
2. GPTQ with act-order: columns in descending diag(H) order, damping 0.03 * mean(diag H), the upper Cholesky factor
   of H^-1 (float64), blocks of 128 columns; each rounding error is spread over the columns not yet rounded. The
   codes are written back in the original column order, so the stored format does not depend on the order.
3. Store OUT/<tensor name>.safetensors: codes U32 [N, K/8] (q + 8 per nibble, low first), row_scales F32 [N] = s,
   row_sums I32 [N] = sum q.

Inputs that share H (gate/up, qkv/z) share one factorization. CPU only (numpy, float64 linear algebra). The full run
takes about 37 minutes with two processes (`--shard 0/2`, `--shard 1/2`, each with OMP_NUM_THREADS=14
VECLIB_MAXIMUM_THREADS=14) and peaks at about 15 GB RSS per process (the 17,408 x 17,408 H of `down`).
Existing output files are kept, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st import save_st  # noqa: E402

KEY = {'mlp.gate_proj': 'mlp_in', 'mlp.up_proj': 'mlp_in', 'mlp.down_proj': 'down_in', 'linear_attn.in_proj_qkv': 'lin_in',
       'linear_attn.in_proj_z': 'lin_in', 'linear_attn.out_proj': 'lin_out_in', 'self_attn.q_proj': 'attn_in',
       'self_attn.o_proj': 'attn_o_in'}
DAMP, ACT_ORDER, GRID, BLOCK = 0.03, True, np.arange(0.70, 1.0001, 0.02), 128


def targets(config=None, layers=None):
    """[(tensor name, module)] in conversion order. 27B: 64 layers, every 4th is full attention."""
    text = (config or {}).get('text_config', config or {})
    n, interval = text.get('num_hidden_layers', 64), text.get('full_attention_interval', 4)
    out = []
    for i in range(n):
        if layers is not None and i not in layers:
            continue
        mods = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        mods += ['self_attn.q_proj', 'self_attn.o_proj'] if (i + 1) % interval == 0 else \
            ['linear_attn.in_proj_qkv', 'linear_attn.in_proj_z', 'linear_attn.out_proj']
        out += [(f'language_model.layers.{i}.{m}', m) for m in mods]
    return out


# ---------------------------------------------------------------- quantization (float32 / float64 numpy)

def absmax_scale(w, ratio=1.0):
    a = np.abs(w).max(axis=1).astype(np.float32)
    return (a * np.float32(ratio)).astype(np.float32) / np.float32(7.0), np.where(a > 0, np.float32(7.0) / np.where(a > 0, (a * np.float32(ratio)).astype(np.float32), 1), np.float32(0)).astype(np.float32)


def mse_scale(w, hdiag):
    a = np.abs(w).max(axis=1).astype(np.float32)
    hd = hdiag.astype(np.float32)
    best = np.full(w.shape[0], np.inf, np.float32)
    br = np.ones(w.shape[0], np.float32)
    for r in GRID:
        s, inv = absmax_scale(w, r)
        q = np.clip(np.rint(w * inv[:, None]), -7, 7)
        e = ((w - q * s[:, None]) ** 2) @ hd
        m = e < best
        best[m] = e[m]
        br[m] = np.float32(r)
    s = (a * br).astype(np.float32) / np.float32(7.0)
    inv = np.where(a > 0, np.float32(7.0) / np.where(a > 0, (a * br).astype(np.float32), 1), np.float32(0)).astype(np.float32)
    return s, inv


def sym_lower_(a, blk=1024):
    """In place: lower triangle := (a + a.T) / 2, block by block (no K x K temporaries)."""
    k = a.shape[0]
    for r0 in range(0, k, blk):
        r1 = min(r0 + blk, k)
        up = a[:r1, r0:r1].T.copy()
        a[r0:r1, :r1] += up
        a[r0:r1, :r1] *= 0.5
    return a


def sym_full_(a, blk=1024):
    """In place a := (a + a.T) / 2, block by block."""
    k = a.shape[0]
    for r0 in range(0, k, blk):
        r1 = min(r0 + blk, k)
        up = a[:r1, r0:r1].T.copy()
        a[r0:r1, :r1] += up
        a[r0:r1, :r1] *= 0.5
        a[:r1, r0:r1] = a[r0:r1, :r1].T
    return a


def factor(hp, frac):
    """Upper Cholesky factor of (H + damping)^-1 as float32; hp (float64) is consumed."""
    dead = np.diag(hp) == 0
    hp[dead, dead] = 1.0
    hp[np.diag_indices_from(hp)] += max(frac * float(np.mean(np.diag(hp))), 1e-12)
    inv = np.linalg.inv(hp)
    del hp
    sym_lower_(inv)
    lower = np.linalg.cholesky(inv)
    del inv
    u = lower.T.astype(np.float32)
    del lower
    if not np.isfinite(u).all():
        raise ValueError('non-finite factor')
    return np.ascontiguousarray(u)


def prepare(h):
    perm = np.argsort(-np.diag(h), kind='stable') if ACT_ORDER else np.arange(h.shape[0])
    return perm, factor(h[np.ix_(perm, perm)] if ACT_ORDER else h.copy(), DAMP)


def compensate(w, s, inv, perm, u):
    n, k = w.shape
    w = w[:, perm].astype(np.float32).copy()
    q = np.empty((n, k), np.int8)
    for l0 in range(0, k, BLOCK):
        l1 = min(l0 + BLOCK, k)
        sub = w[:, l0:l1].copy()
        err = np.empty_like(sub)
        for j in range(l1 - l0):
            c = sub[:, j]
            qq = np.clip(np.rint(c * inv), -7, 7)
            q[:, l0 + j] = qq
            e = (c - qq * s) / u[l0 + j, l0 + j]
            err[:, j] = e
            if j + 1 < l1 - l0:
                sub[:, j + 1:] -= np.outer(e, u[l0 + j, l0 + j + 1:l1])
        if l1 < k:
            w[:, l1:] -= err @ u[l0:l1, l1:]
    out = np.empty_like(q)
    out[:, perm] = q
    return out


def pack(q):
    n, k = q.shape
    v = (q.astype(np.int32) + 8).astype(np.uint32).reshape(n, k // 8, 8)
    out = np.zeros((n, k // 8), np.uint32)
    for j in range(8):
        out |= v[:, :, j] << np.uint32(4 * j)
    return out


def row_sums(q):
    return q.astype(np.int32).sum(axis=1).astype(np.int32)


# ---------------------------------------------------------------- files

def bf16_weight(original, index, name):
    path = original / index[name]
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        spec = json.loads(f.read(size))[name]
    b, e = spec['data_offsets']
    raw = np.fromfile(path, dtype=np.uint16, count=(e - b) // 2, offset=8 + size + b).reshape(spec['shape'])
    return (raw.astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original', type=Path)
    ap.add_argument('gram', help='gram.py output folder(s), comma-separated, searched in order')
    ap.add_argument('out', type=Path)
    ap.add_argument('--layers', help='only these layers, e.g. 0,3 (smoke tests)')
    ap.add_argument('--shard', default='0/1', help='I/N: only layers with layer %% N == I (run N processes)')
    a = ap.parse_args()
    si, sn = map(int, a.shard.split('/'))
    grams = [Path(p) for p in a.gram.split(',')]
    a.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((a.original / 'config.json').read_text())
    index = json.loads((a.original / 'model.safetensors.index.json').read_text())['weight_map']
    layers = {int(x) for x in a.layers.split(',')} if a.layers else None
    cache = {}
    for name, mod in targets(config, layers):
        i = int(name.split('.')[2])
        if i % sn != si:
            continue
        path = a.out / (name + '.safetensors')
        if path.exists():
            continue
        t0 = time.time()
        w = bf16_weight(a.original, index, name + '.weight')
        key = KEY[mod]
        if (i, key) not in cache:   # one factorization per input; the K x K H is freed right after
            src = next((g for g in grams if (g / f'L{i}.{key}.npy').exists()), None)
            if src is None:
                raise SystemExit(f'no H for layer {i} {key} in {a.gram}')
            h = np.load(src / f'L{i}.{key}.npy').astype(np.float64)
            sym_full_(h)
            hd = np.diag(h).copy()
            pu = prepare(h)
            del h
            cache = {(i, key): (hd, pu)}
        hd, pu = cache[(i, key)]
        s, inv = mse_scale(w, hd)
        q = compensate(w, s, inv, *pu)
        tmp = path.with_name(path.name + '.part')
        save_st(tmp, {'codes': pack(q), 'row_scales': s.astype(np.float32), 'row_sums': row_sums(q)})
        tmp.replace(path)
        print(json.dumps({'tensor': name, 'seconds': round(time.time() - t0, 1),
                          'clip7': round(float(np.mean(np.abs(q) == 7)), 5)}), flush=True)


if __name__ == '__main__':
    main()

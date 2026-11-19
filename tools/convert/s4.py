"""Step 7 (27B): row-wise 4-bit files from offset-8 nibbles to signed int4 ("s4", two's complement).

    python tools/convert/s4.py ROWQ4 OUT [--source-label LABEL]

ROWQ4 is the output of rowq4.py: one file per tensor with `codes` (U32 [N, K/8], q + 8 per nibble, low nibble
first), `row_scales` (F32 [N]) and `row_sums` (I32 [N]). XOR 0x88888888 on every packed word turns each nibble
q + 8 into the two's complement of q, which the int8 x int4 tensor-op path of the W4A8 kernel reads directly.
Scales and row sums are copied unchanged. The file metadata records `nibble_encoding: twos_complement` (the engine
refuses files without it) and `source: LABEL/<file name>` (default: ROWQ4 as given). The published files were made
with LABEL `experiments/night-20261119/D-002-full-convert/rowq4`; pass that label to reproduce them byte for byte.
CPU only, about 1 minute for the 368 files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st import header, save_st, tensor  # noqa: E402

PUBLISHED_LABEL = 'experiments/night-20261119/D-002-full-convert/rowq4'


def encode(src, dst, label):
    hdr = header(src)
    d = {k: np.array(tensor(src, k, hdr)) for k in hdr[1] if k != '__metadata__'}
    d['codes'] = np.bitwise_xor(d['codes'], np.uint32(0x88888888))
    save_st(dst, d, metadata={'nibble_encoding': 'twos_complement', 'source': f'{label}/{src.name}'})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rowq4', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--source-label', help='the directory name recorded in each file (default: ROWQ4 as given)')
    a = ap.parse_args()
    label = a.source_label or str(a.rowq4).rstrip('/')
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.rowq4.glob('*.safetensors'))
    for p in files:
        encode(p, a.out / p.name, label)
    print(f'{len(files)} files -> {a.out}')


if __name__ == '__main__':
    main()

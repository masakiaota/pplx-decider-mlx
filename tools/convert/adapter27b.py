"""Step 9 (27B): the Locate LoRA in BF16.

    python tools/convert/adapter27b.py ORIGINAL OUT/adapter.safetensors

The official 27B Locate adapter (ORIGINAL/locate-span/adapter.safetensors) holds 992 FP32 LoRA matrices (A and B,
rank 64) and the FP32 Locate marker embedding. The matrices are cast to BF16 (`astype(mx.bfloat16)`, no scaling);
the marker is copied bit for bit. decider-mlx adds this LoRA at run time on top of the 4-bit weights.
Tensors are written in name order with 8-byte aligned offsets. A few seconds, 1.9 GB of MLX memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault('MLX_ENABLE_TF32', '0')
import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st import header, sha256  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    source = a.original / 'locate-span/adapter.safetensors'
    base, h = header(source)
    specs = {k: v for k, v in h.items() if k != '__metadata__'}
    assert all(v['dtype'] == 'F32' for v in specs.values())
    ab = [k for k in specs if k.endswith(('.lora_a', '.lora_b'))]
    assert len(ab) == 992 and set(specs) - set(ab) == {'language_model.embed_tokens.locate'}
    new, offset = {}, 0
    for k in sorted(specs):
        b, e = specs[k]['data_offsets']
        size = (e - b) // 2 if k in ab else e - b
        assert offset % 8 == 0 and size % 8 == 0
        new[k] = dict(dtype='BF16' if k in ab else 'F32', shape=specs[k]['shape'], data_offsets=[offset, offset + size])
        offset += size
    new['__metadata__'] = {'format': 'pt', 'source_sha256': sha256(source),
                           'conversion': 'A/B F32 to BF16 via MLX astype; marker unchanged'}
    raw = json.dumps(new, separators=(',', ':')).encode()
    raw += b' ' * ((-len(raw)) % 8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('wb') as dst:
        dst.write(struct.pack('<Q', len(raw)))
        dst.write(raw)
        for k in sorted(specs):
            b, e = specs[k]['data_offsets']
            value = np.fromfile(source, dtype=np.float32, count=(e - b) // 4, offset=base + b)
            if k in ab:
                blob = np.asarray(mx.array(value).astype(mx.bfloat16).view(mx.uint16)).tobytes()
            else:
                blob = value.tobytes()
            assert len(blob) == new[k]['data_offsets'][1] - new[k]['data_offsets'][0]
            dst.write(blob)
    print(json.dumps({'out': str(a.out), 'bytes': a.out.stat().st_size,
                      'sha256': hashlib.sha256(a.out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()

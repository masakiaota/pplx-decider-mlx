"""Step 8 (27B): model-rest.safetensors, everything of the 4-bit model that is not replaced by the row-wise 4-bit files.

    python tools/convert/rest.py Q4 OUT

Q4 is the output of `quantize.py --bits 4`. The 368 linear layers that rowq4.py converts (all text linear layers
except the attention k/v projections) are dropped; what remains is the group-64 affine 4-bit k/v projections and
the BF16 tensors (embeddings, norms, convolutions, DeltaNet gates, vision tower), in shard order. They are written to
one file with 8-byte aligned tensors, plus model.safetensors.index.json. CPU only, about 30 s, 3.6 GB.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rowq4 import targets  # noqa: E402
from st import header  # noqa: E402

# The note the published index carries (B013 was the research name of the calibration model, whose k/v and BF16
# tensors are those of Q4).
NOTE = 'B013 minus 368 row-Q4-replaced tensors'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('q4', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((a.q4 / 'config.json').read_text()) if (a.q4 / 'config.json').exists() else None
    replaced = {name for name, _ in targets(config)}
    assert len(replaced) == 368
    index = json.loads((a.q4 / 'model.safetensors.index.json').read_text())
    entries = []
    for shard in sorted(set(index['weight_map'].values())):
        path = a.q4 / shard
        base, h = header(path)
        for name, spec in h.items():
            if name == '__metadata__':
                continue
            b, e = spec['data_offsets']
            stem, suffix = name.rsplit('.', 1)
            if stem in replaced and suffix in ('weight', 'scales', 'biases'):
                continue
            entries.append((name, spec['dtype'], spec['shape'], path, base + b, e - b))
    hdr, off = {}, 0
    for name, dtype, shape, _, _, n in entries:
        hdr[name] = dict(dtype=dtype, shape=shape, data_offsets=[off, off + n])
        off += (n + 7) // 8 * 8
    raw = json.dumps(hdr, separators=(',', ':')).encode()
    raw += b' ' * ((-len(raw)) % 8)
    dst = a.out / 'model-rest.safetensors'
    with dst.open('wb') as out:
        out.write(struct.pack('<Q', len(raw)))
        out.write(raw)
        for _, _, _, src, begin, n in entries:
            out.write(np.fromfile(src, dtype=np.uint8, count=n, offset=begin).tobytes())
            out.write(b'\0' * ((-n) % 8))
    (a.out / 'model.safetensors.index.json').write_text(json.dumps(
        dict(metadata=dict(total_size=off, note=NOTE), weight_map={n: dst.name for n, *_ in entries}), indent=1) + '\n')
    print(json.dumps({'out': str(dst), 'tensors': len(entries), 'bytes': dst.stat().st_size}))


if __name__ == '__main__':
    main()

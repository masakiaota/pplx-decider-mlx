"""Compare a converted folder (or some of its files) with a published manifest.json.

    python tools/convert/verify.py FOLDER MANIFEST [--only PATTERN ...]

MANIFEST is the manifest.json of the Hugging Face repository (or a local copy). Every file the manifest lists
(or those matching --only, a glob on the relative path) is checked by size and sha256. The `mlx_config.json`
files record the MLX version that ran the conversion; when one differs only in `mlx_version`, it is reported as
such. The published row-wise 4-bit files were written by the `safetensors` package, which orders the two metadata
keys of a file arbitrarily (a Rust HashMap); a .safetensors file whose header differs only in that order, with the
same parsed header and byte-identical tensor data, is reported as "metadata order only". Exit status 1 when any
file is missing or differs otherwise.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st import sha256  # noqa: E402


def published_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    value.pop('mlx_version', None)
    return value


def payload_and_header(path):
    with open(path, 'rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(size))
        h = hashlib.sha256()
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return size, hdr, h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', type=Path)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    files = json.loads(a.manifest.read_text())['files']
    names = [f for f in files if not a.only or any(fnmatch.fnmatch(f, p) for p in a.only)]
    reference = a.manifest.parent
    same = differ = version_only = order_only = 0
    for name in names:
        path = a.folder / name
        if not path.exists():
            print(f'missing  {name}')
            differ += 1
            continue
        if path.stat().st_size == files[name]['size'] and sha256(path) == files[name]['sha256']:
            same += 1
            continue
        if name.endswith('mlx_config.json') and (reference / name).exists() and \
                published_json(path) == published_json(reference / name):
            print(f'mlx_version only  {name}')
            version_only += 1
            continue
        if name.endswith('.safetensors') and (reference / name).exists() and \
                payload_and_header(path) == payload_and_header(reference / name):
            order_only += 1
            continue
        print(f'differs  {name}')
        differ += 1
    print(json.dumps({'checked': len(names), 'identical': same, 'mlx_version_only': version_only,
                      'metadata_order_only': order_only, 'different': differ}))
    sys.exit(1 if differ else 0)


if __name__ == '__main__':
    main()

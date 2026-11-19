"""Small safetensors helpers shared by the conversion scripts (numpy and MLX only).

- `header(path)`: (payload offset, parsed JSON header) without reading tensor data.
- `tensor(path, name)`: one tensor as a read-only numpy memmap (BF16 is returned as uint16 bits).
- `save_mlx_aligned(path, tensors)`: `mx.save_safetensors`, then the JSON header padded with spaces to a multiple
  of 8 bytes so every tensor can be memory-mapped by its dtype (the format of all published shards).
- `save_st(path, tensors, metadata)`: the byte layout of the `safetensors` Python package (0.8) for numpy
  arrays: tensors sorted by dtype (widest first) and name, compact JSON, header padded to 8 bytes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

NP = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32, 'U8': np.uint8, 'I8': np.int8, 'U16': np.uint16,
      'I32': np.int32, 'U32': np.uint32, 'I64': np.int64, 'U64': np.uint64}
# safetensors' Dtype order (Rust enum); the package writes higher dtypes first.
ORDER = ['BOOL', 'U8', 'I8', 'I16', 'U16', 'F16', 'BF16', 'I32', 'U32', 'F32', 'F64', 'I64', 'U64']
CODE = {np.dtype(np.uint8): 'U8', np.dtype(np.int8): 'I8', np.dtype(np.uint16): 'U16', np.dtype(np.int16): 'I16',
        np.dtype(np.float16): 'F16', np.dtype(np.int32): 'I32', np.dtype(np.uint32): 'U32', np.dtype(np.float32): 'F32',
        np.dtype(np.float64): 'F64', np.dtype(np.int64): 'I64', np.dtype(np.uint64): 'U64'}


def header(path):
    with open(path, 'rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        return 8 + size, json.loads(f.read(size))


def tensor(path, name, base_header=None):
    base, h = base_header or header(path)
    spec = h[name]
    begin, end = spec['data_offsets']
    return np.memmap(path, mode='r', dtype=NP[spec['dtype']], offset=base + begin, shape=tuple(spec['shape']))


def bf16_to_f32(bits):
    return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32)


def pad_header(raw):
    return raw + b' ' * ((-len(raw)) % 8)


def save_mlx_aligned(mx, path, tensors):
    path = Path(path)
    stage = path.with_name(path.stem + '.stage.safetensors')
    mx.save_safetensors(str(stage), tensors)
    with stage.open('rb') as src, path.open('wb') as dst:
        size = struct.unpack('<Q', src.read(8))[0]
        raw = pad_header(src.read(size))
        dst.write(struct.pack('<Q', len(raw)) + raw)
        shutil.copyfileobj(src, dst, 8 << 20)
    stage.unlink()
    _, h = header(path)
    base = 8 + len(raw)
    assert all((base + v['data_offsets'][0]) % 8 == 0 for k, v in h.items() if k != '__metadata__'), path
    return path


def save_st(path, tensors, metadata=None):
    items = sorted(tensors.items(), key=lambda kv: (-ORDER.index(CODE[kv[1].dtype]), kv[0]))
    hdr, offset = {}, 0
    if metadata is not None:
        hdr['__metadata__'] = metadata
    for name, a in items:
        hdr[name] = {'dtype': CODE[a.dtype], 'shape': list(a.shape), 'data_offsets': [offset, offset + a.nbytes]}
        offset += a.nbytes
    raw = pad_header(json.dumps(hdr, separators=(',', ':')).encode())
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(raw)) + raw)
        for _, a in items:
            f.write(np.ascontiguousarray(a).tobytes())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def clone(src, dst):
    """Copy a file; on macOS an APFS clone (copy-on-write, no extra disk until the copy is modified)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if sys.platform == 'darwin':
        subprocess.run(['cp', '-c', str(Path(src).resolve()), str(dst)], check=True)
    else:
        shutil.copyfile(src, dst)


def json_dump(path, value, indent=2):
    Path(path).write_text(json.dumps(value, indent=indent) + '\n')

"""Experimental read-only safetensors views using public MLX 0.32.4 APIs.

The model files must remain unchanged while arrays exist. Mapped file bytes
remain part of the memory budget, regardless of RSS/footprint accounting.
"""
import json
import math
from pathlib import Path
import struct

import mlx.core as mx
import numpy as np

DTYPES={'BF16':(mx.bfloat16,2),'F16':(mx.float16,2),'F32':(mx.float32,4),
        'U8':(mx.uint8,1),'U32':(mx.uint32,4),'I32':(mx.int32,4),'I64':(mx.int64,8)}

class MappedSafetensors:
    def __init__(self, limit_bytes):
        self.limit_bytes=limit_bytes
        self.original_load=mx.load
        self.files={}
        self.mapped_bytes=0
        self.container_bytes={}

    def __call__(self,path,*args,**kwargs):
        if not isinstance(path,(str,Path)) or Path(path).suffix!='.safetensors':
            return self.original_load(path,*args,**kwargs)
        if args or set(kwargs)-{'return_metadata'}:
            raise ValueError('This experiment accepts only path/return_metadata for safetensors')
        path=Path(path).resolve()
        if path not in self.files:
            stat=path.stat()
            if mx.get_active_memory()+stat.st_size>self.limit_bytes:
                raise MemoryError('Mapped file plus existing MLX active bytes exceed the explicit budget')
            with path.open('rb') as source:
                header_size=struct.unpack('<Q',source.read(8))[0]
                if not 2<=header_size<=min(stat.st_size-8,32_000_000):
                    raise ValueError('Invalid or unexpectedly large safetensors header')
                header=json.loads(source.read(header_size))
            base=8+header_size
            specs={}
            for name,spec in header.items():
                if name=='__metadata__':continue
                dtype,width=DTYPES[spec['dtype']]
                begin,end=spec['data_offsets'];shape=tuple(spec['shape'])
                if not (0<=begin<=end<=stat.st_size-base) or any(s<0 for s in shape):
                    raise ValueError('Invalid tensor bounds')
                if end-begin!=math.prod(shape)*width or (base+begin)%width or (base+end)%width:
                    raise ValueError(f'Tensor requires payload aligned to its dtype: {name}')
                specs[name]=(base+begin,base+end,dtype,shape)
            # Select the widest zero-copy byte container compatible with every
            # tensor boundary. The existing aligned main shards still use U64;
            # Locate's F32 scalars and subsequent offsets require U32.
            boundaries=[stat.st_size,*[offset for spec in specs.values() for offset in spec[:2]]]
            unit=next(size for size in [8,4,2,1] if all(offset%size==0 for offset in boundaries))
            if stat.st_size//unit>2**31-1:
                raise ValueError('No aligned zero-copy container fits the MLX int32 shape limit')
            owner=np.memmap(path,mode='r',dtype={8:np.uint64,4:np.uint32,2:np.uint16,1:np.uint8}[unit])
            raw=mx.asarray(owner,copy=False)
            values={name:raw[begin//unit:end//unit].view(dtype).reshape(shape)
                    for name,(begin,end,dtype,shape) in specs.items()}
            self.container_bytes[str(path)]=unit
            self.files[path]=(owner,raw,values,header.get('__metadata__',{}),
                              (stat.st_ino,stat.st_size,stat.st_mtime_ns))
            self.mapped_bytes+=stat.st_size
        _,_,values,metadata,fingerprint=self.files[path]
        stat=path.stat()
        if fingerprint!=(stat.st_ino,stat.st_size,stat.st_mtime_ns):
            raise RuntimeError('A mapped model file changed during its lifetime')
        return (dict(values),dict(metadata)) if kwargs.get('return_metadata') else dict(values)

    def close(self):
        # Existing MLX views retain the imported NumPy owner through MLX itself.
        self.files.clear()


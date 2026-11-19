"""Step 3 (Flash): the Locate weights, with the official LoRA merged and quantized to affine 8-bit.

    python tools/convert/flash_locate.py ORIGINAL OUT

For every module the official Locate LoRA (ORIGINAL/locate-span/adapter.safetensors, FP32, rank 64, alpha 64)
adapts, the merged weight is

    W' = BF16(FP32(W) + FP32(B) @ FP32(A) * alpha / rank)

from the original BF16 W. The DeltaNet gate projections (in_proj_a, in_proj_b) stay BF16; all other merged
modules are quantized with `mx.quantize(W', bits=8, group_size=64)` (affine). The engine uses these weights only
for Locate questions, in place of the main weights of the same modules, so no LoRA runs at inference time.
The Locate marker embedding (FP32) goes to marker.safetensors; the head and locate_config.json are copied.
MLX_ENABLE_TF32=0 keeps the FP32 product exact. About 10 s, MLX peak about 6.5 GB.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

os.environ['MLX_ENABLE_TF32'] = '0'
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from originals import ORIGINAL  # noqa: E402
from st import save_mlx_aligned, sha256  # noqa: E402

MARKER = 'language_model.embed_tokens.locate'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f'{a.out}: must be empty')
    a.out.mkdir(parents=True, exist_ok=True)
    mx.set_memory_limit(12_000_000_000)
    mx.set_cache_limit(128_000_000)
    original = ORIGINAL['flash']
    folder = a.original / 'locate-span'
    config = json.loads((folder / 'locate_config.json').read_text())
    assert not config['lora'].get('merged'), 'the official adapter is unmerged'
    adapter = mx.load(str(folder / 'adapter.safetensors'))
    mx.eval(adapter)
    modules = sorted(key.removesuffix('.lora_a') for key in adapter if key.endswith('.lora_a'))
    index = json.loads((a.original / 'model.safetensors.index.json').read_text())['weight_map']
    scale = config['lora']['alpha'] / config['lora']['rank']
    quantized, preserved, weight_map, tensor_bytes = [], [], {}, 0
    for shard in sorted({index[name + '.weight'] for name in modules}):
        weights = mx.load(str(a.original / shard))
        converted = {}
        for name in modules:
            if index[name + '.weight'] != shard:
                continue
            base = weights[name + '.weight']
            assert base.dtype == mx.bfloat16, name
            update = adapter[name + '.lora_b'].astype(mx.float32) @ adapter[name + '.lora_a'].astype(mx.float32)
            merged = (base.astype(mx.float32) + update * scale).astype(mx.bfloat16)
            mx.eval(merged)
            if name.endswith(('.in_proj_a', '.in_proj_b')):
                converted[name + '.weight'] = merged
                preserved.append(name)
            else:
                codes, scales, biases = mx.quantize(merged, bits=8, group_size=64, mode='affine')
                mx.eval(codes, scales, biases)
                converted.update({name + '.weight': codes, name + '.scales': scales, name + '.biases': biases})
                quantized.append(name)
        mx.eval(converted)
        save_mlx_aligned(mx, a.out / shard, converted)
        for name, value in converted.items():
            weight_map[name] = shard
            tensor_bytes += value.nbytes
        print(json.dumps({'shard': shard, 'mlx_peak_gb': round(mx.get_peak_memory() / 1e9, 2)}), flush=True)
        del weights, converted, base, update, merged, value
        mx.clear_cache()
    assert adapter[MARKER].dtype == mx.float32
    save_mlx_aligned(mx, a.out / 'marker.safetensors', {MARKER: adapter[MARKER]})
    for name in ('head.safetensors', 'locate_config.json'):
        shutil.copyfile(folder / name, a.out / name)
    metadata = dict(format='decider-locate-merged-mlx-v1', lora_applied=True,
                    base_repository=original['repo'], base_revision=original['revision'],
                    adapter_sha256=sha256(folder / 'adapter.safetensors'),
                    head_sha256=sha256(folder / 'head.safetensors'),
                    merge_formula='BF16(FP32(W) + FP32(B) @ FP32(A) * (alpha/rank))',
                    merge_scale=scale, lora_alpha=config['lora']['alpha'], lora_rank=config['lora']['rank'],
                    merged_modules=modules, quantized_modules=quantized, bf16_merged_modules=preserved,
                    quantization=dict(mode='affine', bits=8, group_size=64), quantizer='mlx.core.quantize',
                    marker_file='marker.safetensors', head_file='head.safetensors', tensor_bytes=tensor_bytes,
                    runtime_contract='Locate-only overlay; no LoRA A/B; unchanged normal Decisions weights',
                    mlx_version=importlib.metadata.version('mlx'), mlx_enable_tf32='0')
    (a.out / 'mlx_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (a.out / 'model.safetensors.index.json').write_text(
        json.dumps(dict(metadata=dict(total_size=tensor_bytes), weight_map=weight_map), indent=2) + '\n')
    print(json.dumps({'out': str(a.out), 'quantized': len(quantized), 'bf16': len(preserved), 'bytes': tensor_bytes}))


if __name__ == '__main__':
    main()

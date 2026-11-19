"""Step 1 (both models): BF16 -> MLX affine group-64 quantization of the text linear layers, shard by shard.

    python tools/convert/quantize.py ORIGINAL OUT --bits 8     # Flash: the base of the shared affine8 weights
    python tools/convert/quantize.py ORIGINAL OUT --bits 4     # 27B: the k/v weights, and the calibration model

Every 2-D `language_model.layers.*.weight` except the DeltaNet gate projections (`in_proj_a`, `in_proj_b`) is
quantized with `mx.quantize(w, group_size=64, bits=B)` (mode affine); everything else (embeddings, norms,
convolutions, gates, the vision tower) is copied as BF16. Names and shard split follow the original checkpoint.
Each shard's JSON header is padded with spaces to a multiple of 8 bytes, so every tensor can be memory-mapped.
Writes OUT/model-*.safetensors, OUT/model.safetensors.index.json and OUT/mlx_config.json, and copies the
tokenizer, configuration files, readout head, LICENSE, NOTICE and locate-span/ of the original, so that OUT is a
complete model folder (the calibration engine of the 27B steps reads it). MLX peak about 7 GB; Flash about 10 s,
27B about 30 s plus reading 54 GB.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MLX_ENABLE_TF32', '0')
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from originals import METADATA, ORIGINAL  # noqa: E402
from st import clone, json_dump, save_mlx_aligned  # noqa: E402

PRESERVED = ['embedding', 'readout', 'vision', 'locate_adapter', 'normalization', 'convolution']


def model_of(folder):
    config = json.loads((Path(folder) / 'config.json').read_text())
    layers = config.get('text_config', config)['num_hidden_layers']
    return {32: 'flash', 64: '27b'}[layers]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--bits', type=int, choices=(4, 8), required=True)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f'{a.out}: must be empty')
    a.out.mkdir(parents=True, exist_ok=True)
    original = ORIGINAL[model_of(a.original)]
    mx.set_memory_limit(12_000_000_000)
    mx.set_cache_limit(256_000_000)
    index = json.loads((a.original / 'model.safetensors.index.json').read_text())
    weight_map, quantized, total = {}, [], 0
    for shard in sorted(set(index['weight_map'].values())):
        weights = mx.load(str(a.original / shard))
        converted = {}
        for name, value in weights.items():
            gate = name.endswith(('.linear_attn.in_proj_a.weight', '.linear_attn.in_proj_b.weight'))
            if name.startswith('language_model.layers.') and name.endswith('.weight') and value.ndim == 2 and not gate:
                packed, scales, biases = mx.quantize(value, group_size=64, bits=a.bits)
                mx.eval(packed, scales, biases)
                prefix = name.removesuffix('.weight')
                converted[name] = packed
                converted[prefix + '.scales'] = scales
                converted[prefix + '.biases'] = biases
                quantized.append(prefix)
            else:
                converted[name] = value
        mx.eval(converted)
        save_mlx_aligned(mx, a.out / shard, converted)
        for name, value in converted.items():
            weight_map[name] = shard
            total += value.nbytes
        print(json.dumps({'shard': shard, 'mlx_peak_gb': round(mx.get_peak_memory() / 1e9, 2)}), flush=True)
        del weights, converted, value
        mx.clear_cache()
    json_dump(a.out / 'model.safetensors.index.json', {'metadata': {'total_size': total}, 'weight_map': weight_map})
    json_dump(a.out / 'mlx_config.json', {
        'format': 'decider-mlx-v1', 'source_repository': original['repo'], 'source_revision': original['revision'],
        'quantization': {'bits': a.bits, 'group_size': 64, 'mode': 'affine'}, 'quantized_modules': quantized,
        'quantize_gates': False, 'preserved': PRESERVED, 'weight_layout': 'official',
        'mlx_version': importlib.metadata.version('mlx')})
    for name in METADATA + ['LICENSE', 'NOTICE']:
        clone(a.original / name, a.out / name)
    for path in sorted((a.original / 'locate-span').iterdir()):
        clone(path, a.out / 'locate-span' / path.name)
    print(json.dumps({'out': str(a.out), 'quantized_modules': len(quantized), 'tensor_bytes': total}))


if __name__ == '__main__':
    main()

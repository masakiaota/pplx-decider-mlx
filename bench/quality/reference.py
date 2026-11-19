"""Produce BF16 reference outputs of the original model over the evaluation sets (same rows as run.py).

    uv run python bench/quality/reference.py --model pplx-decider-flash-v1.3 --out bench/quality/out/flash-bf16.jsonl
    uv run python bench/quality/reference.py --model pplx-decider-v1.3 --original /path/to/pplx-decider-v1.3 \
        --memory-gb 62 --out bench/quality/out/27b-bf16.jsonl

The engine is bf16ref: the original perplexity-ai checkpoint (BF16, unquantized) in plain MLX. Without --original
the checkpoint is downloaded into the Hugging Face cache at the revision below (Flash 9.6 GB, 27B 54.1 GB; set
HF_HUB_OFFLINE=1 to use the cache only).

Memory: MLX peak about 12 GB for Flash and 57 GB for 27B (the whole BF16 model stays resident). 27B therefore
needs a Mac with 96 GB or more in practice, with --memory-gb 62. Run nothing else on the GPU meanwhile.
--memory-gb sets MLX's memory limit; an input whose estimated peak would exceed it fails with MemoryError (recorded
as an error row) instead of swapping. --max-length 12000 (tokens per sequence) covers every evaluation set.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import bf16ref  # noqa: E402  (imports decider_mlx before mlx)

from common import DATA, jl, log, run_suite, suite_files  # noqa: E402

ORIGINALS = {   # official name: (Hugging Face repository, pinned revision)
    'pplx-decider-flash-v1.3': ('perplexity-ai/pplx-decider-flash-v1.3', 'd06fb3f2d97e0a52325a0d463c450cf0338f44fe'),
    'pplx-decider-v1.3': ('perplexity-ai/pplx-decider-v1.3', '9c2ebb27470d3dec2ad7ab727727e40d142c7213'),
}
DEFAULT_MEMORY_GB = {'pplx-decider-flash-v1.3': 48, 'pplx-decider-v1.3': 62}


def original_snapshot(model):
    from huggingface_hub import snapshot_download
    repo, revision = ORIGINALS[model]
    return Path(snapshot_download(repo, revision=revision,
                                  allow_patterns=['*.json', '*.jinja', '*.safetensors', 'locate-span/*']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=sorted(ORIGINALS))
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--original', type=Path, help='local folder with the original checkpoint (default: download)')
    ap.add_argument('--memory-gb', type=float, help='MLX memory limit (default: 48 for Flash, 62 for 27B)')
    ap.add_argument('--max-length', type=int, default=12000)
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--sets', default='all', help="core, extra, all, or comma-separated set names (default: all)")
    ap.add_argument('--limit', type=int, default=0, help='only the first N lines of each file (smoke test)')
    ap.add_argument('--ids', type=Path, help='only the ids in this JSONL file (lines with "id")')
    ap.add_argument('--locate-threshold', type=float, help='keep Locate spans with score >= this (default: 0.5)')
    ap.add_argument('inputs', nargs='*', help='evaluation JSONL files (default: --sets in DATA/eval)')
    a = ap.parse_args()
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        raise SystemExit('MLX_ENABLE_TF32 must be 0 (import decider_mlx before mlx)')
    files = suite_files(a.inputs, a.data, a.sets)
    ids = {r['id'] for r in jl(a.ids)} if a.ids else None
    path = a.original or original_snapshot(a.model)
    started = time.perf_counter()
    engine = bf16ref.Engine(path, max_length=a.max_length, memory_limit_gb=a.memory_gb or DEFAULT_MEMORY_GB[a.model])
    if engine.model_name != a.model:
        raise SystemExit(f'{path} holds {engine.model_name}, not {a.model}')
    import mlx.core as mx
    log(event='loaded', model=a.model, path=str(path), seconds=time.perf_counter() - started,
        active=mx.get_active_memory(), peak=mx.get_peak_memory())
    if a.locate_threshold is not None:
        locate = engine.load_locate()
        locate.config['thresholds'] = {**locate.config.get('thresholds', {}), 'default': a.locate_threshold}
        log(event='locate_threshold', value=a.locate_threshold)

    def decide(request, mode, temperature):
        return engine.decide(request, execution_mode=mode, temperature=temperature)

    run_suite(decide, files, a.out, f'bf16ref {a.model}', a.limit, ids, peak=mx.get_peak_memory)


if __name__ == '__main__':
    main()

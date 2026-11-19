"""Run a decider-mlx build over the evaluation sets and write one output row per request.

    uv run python bench/quality/run.py --model pplx-decider-flash-v1.3 --out bench/quality/out/flash.jsonl
    uv run python bench/quality/run.py --model pplx-decider-v1.3 --out out/27b.jsonl --sets extra
    uv run python bench/quality/run.py --model pplx-decider-flash-v1.3 --out x.jsonl path/to/set.jsonl ...

The engine is the installed decider_mlx package (the working tree under `uv run`), with the weights from the
Hugging Face cache, or from --weights (a local copy of the converted weights repository, for example one you built
with tools/convert before uploading it). Each request is sent alone with its line's options (execution mode,
temperature 1), as `app.decide(request, execution_mode=..., temperature=...)`.

Output row: {"id", "set", "engine", "response" | "error", "elapsed_s", "request_sha256"}. The run resumes: ids
already in --out are skipped. --locate-threshold 0.2 keeps Locate spans down to that score for locate_soft.py
(decisions are unaffected; compare.py --locate-min-score 0.5 then gives the normal Locate output back).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import decider_mlx  # noqa: E402  (before mlx)

from common import DATA, jl, log, run_suite, suite_files  # noqa: E402

LOCATE_WARMUP = {"state": "The Treaty of Paris was signed in 1783.",
                 "questions": {"a": {"type": "locate", "instructions": "When was the treaty signed?"}}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=sorted(decider_mlx.MODELS))
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--weights', type=Path, help='local copy of the converted weights repository (default: HF cache)')
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--sets', default='all', help="core, extra, all, or comma-separated set names (default: all)")
    ap.add_argument('--limit', type=int, default=0, help='only the first N lines of each file (smoke test)')
    ap.add_argument('--ids', type=Path, help='only the ids in this JSONL file (lines with "id")')
    ap.add_argument('--locate-threshold', type=float, help='keep Locate spans with score >= this (default: 0.5)')
    ap.add_argument('inputs', nargs='*', help='evaluation JSONL files (default: --sets in DATA/eval)')
    a = ap.parse_args()
    files = suite_files(a.inputs, a.data, a.sets)
    ids = {r['id'] for r in jl(a.ids)} if a.ids else None
    started = time.perf_counter()
    internal = {}
    if a.weights:
        from decider_mlx.registry import weights_from_snapshot
        internal['_weights'] = weights_from_snapshot(a.model, a.weights)
    app = decider_mlx.load(a.model, **internal)
    app.wait_ready()   # the startup preparation runs on the same worker; finish it first for clean timings
    import mlx.core as mx
    log(event='loaded', model=a.model, version=decider_mlx.__version__, seconds=time.perf_counter() - started)
    if a.locate_threshold is not None:
        app.decide(LOCATE_WARMUP, temperature=1.0)       # loads the Locate weights
        locate = app.engine.locate
        locate.config['thresholds'] = {**locate.config.get('thresholds', {}), 'default': a.locate_threshold}
        log(event='locate_threshold', value=a.locate_threshold)

    def decide(request, mode, temperature):
        with contextlib.redirect_stdout(sys.stderr):
            return app.decide(request, execution_mode=mode, temperature=temperature)

    run_suite(decide, files, a.out, f'decider-mlx {decider_mlx.__version__} {a.model}', a.limit, ids,
              peak=mx.get_peak_memory)


if __name__ == '__main__':
    main()

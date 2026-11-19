"""Shared pieces of the quality tools: dataset sources, the evaluation suite, JSONL helpers and the runner loop.

No model code and no third-party imports here, so download/build/check/compare run without MLX.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'          # default working folder (ignored by git): raw/, eval/, calib/
USER_AGENT = 'decider-mlx-quality-bench/1'

# Public sources, pinned. (local file name in raw/): (dataset repo, revision, file in the repo, sha256)
PARQUET = {
    'boolq-validation.parquet': ('google/boolq', '35b264d03638db9f4ce671b711558bf7ff0f80d5',
                                 'data/validation-00000-of-00001.parquet',
                                 '52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8'),
    'boolq-train.parquet': ('google/boolq', '35b264d03638db9f4ce671b711558bf7ff0f80d5',
                            'data/train-00000-of-00001.parquet',
                            '4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623'),
    'squad_v2-validation.parquet': ('rajpurkar/squad_v2', '3ffb306f725f7d2ce8394bc1873b24868140c412',
                                    'squad_v2/validation-00000-of-00001.parquet',
                                    '0560174ab095c5ac0a8c8dc8da05f1625453c45a77e4ce9cabc6947ddfdd24cb'),
    'squad_v2-train.parquet': ('rajpurkar/squad_v2', '3ffb306f725f7d2ce8394bc1873b24868140c412',
                               'squad_v2/train-00000-of-00001.parquet',
                               'f6da32ffb482ff463ad056477740d1bb284b96a45db3a08bee6a225ca6abf291'),
    # Qasper: the Hub's Parquet conversion (refs/convert/parquet) at this commit
    'qasper-validation.parquet': ('allenai/qasper', '06806e4608976fc2fac0a090ac425d5b2b29caf4',
                                  'qasper/validation/0000.parquet',
                                  '089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222'),
    'qasper-train.parquet': ('allenai/qasper', '06806e4608976fc2fac0a090ac425d5b2b29caf4',
                             'qasper/train/0000.parquet',
                             '9af08092ee26c4f700202c1f90d1592b662926f23f3a308a10ff0a53345e37fe'),
}
# VQAv2 rows come from the Hugging Face dataset viewer API (it has no pinned revision; the build checks image sha256).
VQA_REPO, VQA_REVISION = 'lmms-lab-encoder/VQAv2', '32665d35052eb4a6d4414851c3c829a72754915a'
VQA_PAGES = [(0, 20), (1000, 100), (1100, 100), (1200, 100), (1300, 100), (1400, 100), (1500, 100),
             (100000, 60), (100060, 100)]   # (offset, length) of validation rows; raw/vqa-rows-<offset>.json

# The evaluation suite in run order. The first eleven files are the core suite (772 requests); the last three are the
# additional sets that raise the resolution for Locate, long inputs and images.
CORE = ['boolq100', 'boolq-val300', 'boolq-choice-score60', 'squad-choice60', 'squad-locate100',
        'squad-multi30-isolated', 'squad-multi30-separate', 'vqa30', 'qasper-long30',
        'features16-separate', 'features16-isolated']
EXTRA = ['squad-locate200b', 'qasper-long40b', 'vqa60b']
CALIB = ['calib-boolq-train2000', 'calib-squad-train500', 'calib-squad-locate-train200', 'calib-qasper-train20',
         'calib-vqa20']


def jl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_sha256(request):
    """Identity of a request as written in the evaluation file (image paths as written, not resolved)."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def suite_files(inputs, data, sets):
    """Evaluation files to run: explicit paths, or the named sets in data/eval (core, extra, all, or a,b,c)."""
    if inputs:
        return [Path(p) for p in inputs]
    names = {'core': CORE, 'extra': EXTRA, 'all': CORE + EXTRA}.get(sets) or sets.split(',')
    files = [Path(data) / 'eval' / f'{n}.jsonl' for n in names]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f'missing evaluation files (run build_sets.py first): {", ".join(missing)}')
    return files


def resolve_images(request, folder):
    """Image paths in the evaluation files are relative to the file's folder; data URLs and absolute paths pass."""
    images = request.get('images')
    if not images:
        return request
    out = []
    for v in images:
        if isinstance(v, str) and not v.startswith('data:') and not Path(v).is_absolute():
            v = str(Path(folder) / v)
        out.append(v)
    return {**request, 'images': out}


def log(**event):
    print(json.dumps(event), file=sys.stderr, flush=True)


def run_suite(decide, files, out, engine_name, limit=0, ids=None, peak=None):
    """Run decide(request, execution_mode, temperature) over every line and append output rows to `out`.

    Output row: {"id", "set", "engine", "response" | "error", "elapsed_s", "request_sha256"}. Resumable: ids already
    in `out` are skipped. A failing request is recorded as "error" (type: message), never hidden.
    """
    out = Path(out)
    done = {r['id'] for r in jl(out)} if out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    started, n = time.perf_counter(), 0
    with open(out, 'a') as sink:
        for path in files:
            lines = jl(path)
            if limit:
                lines = lines[:limit]
            for rec in lines:
                if rec['id'] in done or (ids is not None and rec['id'] not in ids):
                    continue
                opts = rec.get('options') or {}
                request = resolve_images(rec['request'], Path(path).parent)
                t = time.perf_counter()
                try:
                    resp = decide(request, opts.get('execution_mode', 'separate'), opts.get('temperature', 1.0))
                    row = dict(id=rec['id'], set=rec.get('set'), engine=engine_name, response=resp,
                               elapsed_s=time.perf_counter() - t)
                except Exception as e:  # noqa: BLE001 - recorded, not hidden
                    row = dict(id=rec['id'], set=rec.get('set'), engine=engine_name, error=f'{type(e).__name__}: {e}')
                row['request_sha256'] = request_sha256(rec['request'])
                sink.write(json.dumps(row, ensure_ascii=False) + '\n')
                sink.flush()
                n += 1
                if n % 25 == 0:
                    log(event='progress', n=n, elapsed=time.perf_counter() - started, peak=peak() if peak else None)
    log(event='done', n=n, seconds=time.perf_counter() - started, peak=peak() if peak else None)
    return n

"""Check that calibration data does not overlap the evaluation sets. Exits 1 on any overlap.

    python bench/quality/check_sets.py [--data bench/quality/data] [EXTRA.jsonl ...]

Checks every DATA/calib/*.jsonl, plus any JSONL given (your own calibration or development data: lines with
"request" (a request or a list of them) or bare requests), against every DATA/eval/*.jsonl:

- exact keys: normalized state text (whitespace collapsed, lower case), normalized question text (the generic
  "Choose the best matching option." is ignored), (dataset, split, index), image id and image sha256, Qasper paper id;
- near duplicates: an evaluation passage contained in a calibration state or the other way round, and a shared
  120-character prefix (BoolQ has longer and shorter versions of the same paragraph in train and validation).
Also checks that evaluation ids are unique. Writes DATA/check.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from common import DATA, jl, sha256_file

VQA_STATE = 'Answer the question about the image.'   # the same for every VQA row: not a key
norm = lambda s: re.sub(r'\s+', ' ', s).strip().lower()
h = lambda s: hashlib.sha256(norm(s).encode()).hexdigest()
GENERIC = h('Choose the best matching option.')


def requests(rec):
    r = rec.get('request', rec)
    return r if isinstance(r, list) else [r]


def state_text(req):
    st = req.get('state')
    return st if isinstance(st, str) else json.dumps(st, ensure_ascii=False, sort_keys=True)


def keys(rec):
    out = set()
    for req in requests(rec):
        if not isinstance(req, dict) or 'state' not in req:
            continue
        st = state_text(req)
        if st != VQA_STATE:
            out.add(('state', h(st)))
        for q in (req.get('questions') or {}).values():
            if isinstance(q, dict) and q.get('instructions') and h(q['instructions']) != GENERIC:
                out.add(('question', h(q['instructions'])))
    m = rec.get('meta') or {}
    for k in ('image_id', 'paper'):
        if k in m:
            out.add((k, str(m[k])))
    if 'image_sha256' in m:
        out.add(('image', m['image_sha256']))
    if m.get('dataset'):
        out.add(('source', json.dumps((m.get('dataset'), m.get('split'),
                                       m.get('index', m.get('id', m.get('row_idx', m.get('question_id'))))))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('files', nargs='*', help='more calibration or development JSONL files to check')
    a = ap.parse_args()
    evals = {p.stem: jl(p) for p in sorted((a.data / 'eval').glob('*.jsonl'))}
    calib = {p.stem: jl(p) for p in sorted((a.data / 'calib').glob('*.jsonl'))}
    calib.update({Path(f).stem: jl(f) for f in a.files})
    if not evals:
        raise SystemExit(f'no evaluation files in {a.data / "eval"}')

    ekeys, states = {}, {}
    for name, rs in evals.items():
        for r in rs:
            for k in keys(r):
                ekeys.setdefault(k, set()).add(name)
            for req in requests(r):
                st = state_text(req)
                if st != VQA_STATE:
                    states.setdefault(norm(st), set()).add(name)
    long_states = [s for s in states if len(s) > 200]
    prefixes = {}
    for s in long_states:
        prefixes.setdefault(s[:120], set()).update(states[s])

    overlap = {}
    for name, rs in calib.items():
        hits = []
        for r in rs:
            why = {}
            for k in keys(r):
                if k in ekeys:
                    why[k[0]] = sorted(ekeys[k])
            for req in requests(r):
                if not isinstance(req, dict) or 'state' not in req:
                    continue
                ns = norm(state_text(req))
                if ('state' in why) or state_text(req) == VQA_STATE:
                    continue
                sub = [s for s in long_states if s in ns or (len(ns) > 200 and ns in s)]
                if sub:
                    why['contains_or_inside_eval_passage'] = sorted(set().union(*(states[s] for s in sub)))
                elif len(ns) > 200 and ns[:120] in prefixes:
                    why['same_120char_prefix'] = sorted(prefixes[ns[:120]])
            if why:
                hits.append(dict(id=r.get('id'), why=why))
        overlap[name] = hits
    ids = [r['id'] for rs in evals.values() for r in rs]
    report = dict(eval_counts={k: len(v) for k, v in evals.items()}, calib_counts={k: len(v) for k, v in calib.items()},
                  disjoint=all(not v for v in overlap.values()), overlap=overlap, duplicate_eval_ids=len(ids) - len(set(ids)),
                  sha256={f'{d}/{p.name}': sha256_file(p) for d in ('eval', 'calib') for p in sorted((a.data / d).glob('*.jsonl'))})
    (a.data / 'check.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    for name, hits in overlap.items():
        print(f'{name:32s} {len(calib[name]):5d} lines, {len(hits)} overlapping' + (f' e.g. {hits[0]}' if hits else ''))
    print(f'disjoint: {report["disjoint"]}, duplicate evaluation ids: {report["duplicate_eval_ids"]}')
    sys.exit(0 if report['disjoint'] and not report['duplicate_eval_ids'] else 1)


if __name__ == '__main__':
    main()

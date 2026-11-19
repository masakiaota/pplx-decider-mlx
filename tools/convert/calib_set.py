"""Step 4a (27B): the calibration requests, from public training data that do not overlap any evaluation set.

    python tools/convert/calib_set.py OUT [--data bench/quality/data]

Needs the files of bench/quality (`download.py`, then `build_sets.py`; see bench/quality/README.md) and pyarrow
(`uv run --with pyarrow ...`). CPU only, a few seconds. Labels are never read. Writes:

- OUT/stage1.jsonl: the 4 requests the calibration model (calib_model.py) is calibrated on. These are the first
  four requests of the boolq100 gate set: an early choice, kept so that the published weights can be reproduced.
  Their influence on the final weights is indirect (they only shape the inputs from which H is collected).
- OUT/mix.jsonl: 797 requests, about 240K tokens, in three segments (field "seg"), run in this order by gram.py:
  A: 307 BoolQ train passages with one of their questions; every fifth item instead joins 6 unrelated passages
     (about 1.6K tokens) and asks the question of one of them. BoolQ train rows whose passage appears in BoolQ
     validation are dropped (15% of train), one question per passage, shuffled with a fixed seed; the first 310
     items, minus 3 that the overlap check below flags.
  B: the first 300 SQuAD v2 train questions and the first 10 Qasper train papers of the calibration pool
     (bench/quality data/calib), interleaved by length.
  C: the first 180 SQuAD v2 train Locate requests of the calibration pool.
- OUT/pool.jsonl and OUT/heldout.jsonl: the full BoolQ pool (520) and 30 held-out items with disjoint passages
  (for measuring layer error).

Overlap check: every request is compared with the core evaluation sets (data/eval) by normalized state and
question text, an evaluation passage contained in a calibration state or the other way round, and a shared
120-character prefix. The script fails if mix.jsonl still overlaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ['boolq100', 'boolq-val300', 'boolq-choice-score60', 'squad-choice60', 'squad-locate100',
        'squad-multi30-isolated', 'squad-multi30-separate', 'vqa30', 'qasper-long30',
        'features16-separate', 'features16-isolated']
SEED = 20261119
SEGMENT_A = 310          # BoolQ pool items in segment A: the first 310 reach 100K tokens (counted at layer 0)
VQA_STATE = 'Answer the question about the image.'
norm = lambda s: re.sub(r'\s+', ' ', s).strip().lower()
GENERIC = {norm('Choose the best matching option.')}


def jl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))


def requests(rec):
    r = rec.get('request', rec)
    return r if isinstance(r, list) else [r]


def texts(req):
    st = req.get('state')
    st = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False, sort_keys=True)
    qs = [q.get('instructions') for q in (req.get('questions') or {}).values() if isinstance(q, dict) and q.get('instructions')]
    return st, qs


class Overlap:
    """Evaluation texts of the core suite; `ids(rows)` returns the ids of overlapping calibration rows."""

    def __init__(self, eval_dir):
        self.states, self.questions = {}, {}
        for name in CORE:
            for rec in jl(Path(eval_dir) / f'{name}.jsonl'):
                for req in requests(rec):
                    st, qs = texts(req)
                    if st != VQA_STATE:
                        self.states.setdefault(norm(st), set()).add(name)
                    for q in qs:
                        if norm(q) not in GENERIC:
                            self.questions.setdefault(norm(q), set()).add(name)
        self.long = [s for s in self.states if len(s) > 200]
        self.prefixes = {s[:120] for s in self.long}

    def ids(self, rows):
        out = set()
        for rec in rows:
            for req in requests(rec):
                if not isinstance(req, dict) or 'state' not in req:
                    continue
                st, qs = texts(req)
                ns = norm(st)
                hit = ns in self.states or any(s in ns or (len(ns) > 200 and ns in s) for s in self.long) \
                    or (len(ns) > 200 and ns[:120] in self.prefixes) or any(norm(q) in self.questions for q in qs)
                if hit:
                    out.add(rec.get('id'))
        return out


def boolq_pool(data):
    import pyarrow.parquet as pq
    train = pq.read_table(data / 'raw/boolq-train.parquet', columns=['question', 'passage']).to_pylist()
    val_passages = {r['passage'] for r in pq.read_table(data / 'raw/boolq-validation.parquet', columns=['passage']).to_pylist()}
    eval_states = {r['request']['state'] for r in jl(data / 'eval/boolq100.jsonl')}
    rows = [dict(i=i, **r) for i, r in enumerate(train) if r['passage'] not in val_passages]
    seen, uniq = set(), []
    for r in rows:                       # one question per passage
        if r['passage'] not in seen:
            seen.add(r['passage'])
            uniq.append(r)
    rng = random.Random(SEED)
    rng.shuffle(uniq)

    def item(kind, parts, q):
        return dict(id=f"boolq-train-{kind}-{'-'.join(str(p['i']) for p in parts)}", kind=kind,
                    source_indices=[p['i'] for p in parts],
                    request=dict(state='\n\n'.join(p['passage'] for p in parts),
                                 questions={'answer': {'type': 'noul', 'instructions': q['question']}}))

    def make(src, n):
        out, it = [], iter(src)
        for k in range(n):
            if k % 5 == 4:
                parts = [next(it) for _ in range(6)]
                out.append(item('long6', parts, parts[rng.randrange(6)]))
            else:
                p = next(it)
                out.append(item('single', [p], p))
        return out

    heldout = make(uniq[:200], 30)
    pool = make(uniq[sum(len(x['source_indices']) for x in heldout):], 520)
    for x in heldout + pool:
        assert x['request']['state'] not in eval_states
        assert not any(s in val_passages for s in x['request']['state'].split('\n\n'))
    return pool, heldout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path)
    ap.add_argument('--data', type=Path, default=ROOT / 'bench/quality/data')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    overlap = Overlap(a.data / 'eval')
    pool, heldout = boolq_pool(a.data)
    write(a.out / 'pool.jsonl', pool)
    write(a.out / 'heldout.jsonl', heldout)
    bad = overlap.ids(pool)

    def calib(name):
        return [dict(id=r['id'], request=r['request']) for r in jl(a.data / 'calib' / name)]
    squad, qasper, loc = calib('calib-squad-train500.jsonl'), calib('calib-qasper-train20.jsonl'), calib('calib-squad-locate-train200.jsonl')
    seg_a = [dict(id=x['id'], request=x['request'], seg='A') for x in pool[:SEGMENT_A] if x['id'] not in bad]
    seg_b, streams, pos, chars = [], [squad[:300], qasper[:10]], [0, 0], [0, 0]
    total = [sum(len(x['request']['state']) for x in s) for s in streams]
    while any(p < len(s) for p, s in zip(pos, streams)):
        k = min((i for i in range(2) if pos[i] < len(streams[i])), key=lambda i: chars[i] / total[i])
        x = streams[k][pos[k]]
        pos[k] += 1
        chars[k] += len(x['request']['state'])
        seg_b.append(dict(x, seg='B'))
    seg_c = [dict(x, seg='C') for x in loc[:180]]
    mix = seg_a + seg_b + seg_c
    assert not overlap.ids(mix), 'calibration overlaps the evaluation sets'
    write(a.out / 'mix.jsonl', mix)
    write(a.out / 'locate-heldout.jsonl', loc[180:])
    stage1 = [dict(id=r['id'], request=r['request']) for r in jl(a.data / 'eval/boolq100.jsonl')[:4]]
    write(a.out / 'stage1.jsonl', stage1)
    print(json.dumps({'mix': len(mix), 'segments': [len(seg_a), len(seg_b), len(seg_c)], 'dropped': sorted(bad & {x['id'] for x in pool[:SEGMENT_A]}),
                      'mix_sha256': hashlib.sha256((a.out / 'mix.jsonl').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()

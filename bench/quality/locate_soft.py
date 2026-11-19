"""Continuous Locate comparison of runs made with --locate-threshold T (run.py, reference.py): span scores as numbers.

    python bench/quality/locate_soft.py --ref REF_SOFT.jsonl --cand CAND_SOFT.jsonl [--floor 0.2]
        [--eval DATA/eval/squad-locate100.jsonl ...] [--hard CAND_NORMAL.jsonl] [--cand-b B_SOFT.jsonl] [--out x.json]

Spans are matched by (start, end). A span missing on one side is censored at T (its score is < T there), so the
difference is a lower bound. --hard checks that cutting the candidate's soft output at 0.5 reproduces its normal run.
"""
import argparse, json
from pathlib import Path

from compare import f1, squad_norm


def load(p):
    out = {}
    for l in Path(p).read_text().splitlines():
        r = json.loads(l)
        if 'error' in r or not isinstance(r.get('response'), dict):
            continue
        for k, a in r['response']['answers'].items():
            if a['type'] == 'locate':
                out[(r['id'], k)] = {(s['start'], s['end']): (s['score'], s['text']) for s in a['spans']}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True); ap.add_argument('--cand', required=True)
    ap.add_argument('--floor', type=float, default=0.2); ap.add_argument('--eval', nargs='*', default=[])
    ap.add_argument('--hard', help='candidate normal-threshold output: check that soft>=0.5 reproduces it')
    ap.add_argument('--cand-b', help='second candidate: paired bootstrap of per-question mean|dscore| (cand - cand_b)')
    ap.add_argument('--out')
    a = ap.parse_args()
    R, C = load(a.ref), load(a.cand)
    gold = {}
    for e in a.eval:
        for l in Path(e).read_text().splitlines():
            r = json.loads(l)
            if isinstance(r.get('label'), list):
                gold[r['id']] = [g['text'] for g in r['label']]
    T = a.floor
    perq = {}
    signed = []
    d, flips, top_same, n, both_empty = [], 0, 0, 0, 0
    lo = lambda x: max(x, T) if x is not None else T
    gf = {'ref': [], 'cand': []}
    near = 0
    for key in sorted(set(R) & set(C)):
        r, c = R[key], C[key]
        n += 1
        spans = set(r) | set(c)
        if not spans:
            both_empty += 1
        qd = []
        for s in spans:
            rs, cs = (r[s][0] if s in r else None), (c[s][0] if s in c else None)
            if rs is not None and cs is not None:
                qd.append(abs(rs - cs)); signed.append(cs - rs)
            else:
                qd.append(abs(lo(rs) - lo(cs)))  # censored lower bound of the difference
                signed.append(lo(cs) - lo(rs))
            if (lo(rs) >= 0.5) != (lo(cs) >= 0.5):
                flips += 1
            if abs(lo(rs) - 0.5) < 0.1:
                near += 1
        d.extend(qd)
        perq[key] = sum(qd) / len(qd) if qd else 0.0
        top = lambda x: max(x.items(), key=lambda kv: kv[1][0])[0] if x else None
        top_same += top(r) == top(c)
        if key[0] in gold:
            for who, x in (('ref', r), ('cand', c)):
                hard = {s: v for s, v in x.items() if v[0] >= 0.5}
                pred = max(hard.values())[1] if hard else ''
                g = gold[key[0]]
                gf[who].append(float(pred == '') if not g else max(f1(pred, t) for t in g))
    rep = dict(floor=T, questions=n, both_empty=both_empty, spans_compared=len(d), mean_abs_dscore=sum(d) / max(1, len(d)),
               max_abs_dscore=max(d, default=0), mean_signed_dscore=sum(signed) / max(1, len(signed)),
               mean_per_question_abs_dscore=sum(perq.values()) / max(1, len(perq)), decision_flips_at_05=flips, ref_spans_near_05=near, top_span_same=top_same,
               gold_f1={k: sum(v) / max(1, len(v)) for k, v in gf.items()})
    if a.cand_b:
        import random
        B = load(a.cand_b)
        pb = {}
        for key in perq:
            if key in B:
                r, c = R[key], B[key]
                qd = [abs(lo(r[s][0] if s in r else None) - lo(c[s][0] if s in c else None)) if not (s in r and s in c) else abs(r[s][0] - c[s][0]) for s in set(r) | set(c)]
                pb[key] = sum(qd) / len(qd) if qd else 0.0
        keys = sorted(set(perq) & set(pb))
        diffs = [perq[k] - pb[k] for k in keys]
        rng = random.Random(0)
        means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
        rep['paired_vs_b'] = dict(cand_b=a.cand_b, n=len(keys), diff=sum(diffs) / len(diffs), ci95=[means[50], means[1949]])
    if a.hard:
        H = load(a.hard)
        mism = sum({s for s, v in C[k].items() if v[0] >= 0.5} != set(H[k]) for k in set(C) & set(H))
        rep['hard_reproduction_mismatches'] = mism
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()

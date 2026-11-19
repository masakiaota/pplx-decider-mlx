"""Compare a build's outputs with reference outputs of the original model (and with the gold labels).

    python bench/quality/compare.py --ref out/flash-bf16.jsonl --cand out/flash.jsonl --out report.json
    python bench/quality/compare.py --ref REF.jsonl --cand A.jsonl --cand-b B.jsonl   # paired A vs B, bootstrap CI

--eval defaults to every file in DATA/eval (gold labels, set names, request identity). Output files are JSONL
lines with "id" and "response" ({"answers": {...}}) or "error", as written by run.py and reference.py; any file of
that shape works. A candidate row whose request_sha256 differs from the evaluation set's request is counted as an
error, not compared. Metrics per set and per question type (noul / choice / score / locate), see README.md.
"""
import argparse, hashlib, json, math, random, sys
from pathlib import Path

from common import DATA, jl

BOUNDARY = 0.1   # |p_ref - 0.5| < 0.1
EPS = 1e-7


def load_outputs(spec):
    """-> {id: answers dict}, {id: set}"""
    out, sets = {}, {}
    for r in jl(spec):
        if r.get('request_sha256'):
            REQ_HASH.setdefault(spec, {})[r['id']] = r['request_sha256']
        if 'error' in r:
            out[r['id']] = {'__error__': r['error']}
        elif isinstance(r['response'], list):  # array request: one entry per element
            for j, resp in enumerate(r['response']):
                out[f"{r['id']}#{j}"] = resp['answers']
        else:
            out[r['id']] = r['response']['answers']
        if r.get('set'):
            sets[r['id']] = r['set']
    return out, sets


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def kl(p, q):
    return sum(a * math.log(max(a, EPS) / max(b, EPS)) for a, b in zip(p, q) if a > 0)


def probs(ans):
    if ans['type'] == 'noul':
        return [1 - ans['noul'], ans['noul']]
    return list(ans['probabilities'].values())


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


REQ_HASH = {}  # output file -> {id: sha256 of the request it was run on}
LOCATE_MIN = 0.0  # set by --locate-min-score (0.5 turns a --locate-threshold run back into normal output)


def span_set(ans):
    return {(s['start'], s['end']) for s in ans.get('spans', []) if s['score'] >= LOCATE_MIN}


def chars(spans):
    return {c for a, b in spans for c in range(a, b)}


def squad_norm(t):
    import re, string
    t = t.lower()
    t = ''.join(ch for ch in t if ch not in set(string.punctuation))
    t = re.sub(r'\b(a|an|the)\b', ' ', t)
    return ' '.join(t.split())


def f1(pred, gold):
    p, g = squad_norm(pred).split(), squad_norm(gold).split()
    if not p or not g:
        return float(p == g)
    from collections import Counter
    c = sum((Counter(p) & Counter(g)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)


def compare_item(key, r, c, label):
    """one question: returns a row dict"""
    t = r['type']
    row = dict(key=key, type=t)
    if t == 'locate':
        rs, cs = span_set(r), span_set(c)
        rc, cc = chars(rs), chars(cs)
        row.update(span_exact=rs == cs, empty_agree=(not rs) == (not cs),
                   char_jaccard=1.0 if not rc and not cc else len(rc & cc) / len(rc | cc),
                   ref_spans=len(rs), cand_spans=len(cs))
        kept = lambda a: [s for s in a.get('spans', []) if s['score'] >= LOCATE_MIN]
        top = lambda a: max(kept(a), key=lambda s: s['score'])['text'] if kept(a) else ''
        row['top_text_equal'] = top(r) == top(c)
        rsc = {(s['start'], s['end']): s['score'] for s in kept(r)}
        csc = {(s['start'], s['end']): s['score'] for s in kept(c)}
        shared = rs & cs
        row['shared_span_max_dscore'] = max((abs(rsc[k] - csc[k]) for k in shared), default=0.0)
        diff = [rsc[k] for k in rs - cs] + [csc[k] for k in cs - rs]
        row['ref_only_spans'], row['cand_only_spans'] = len(rs - cs), len(cs - rs)
        row['boundary_span_flip'] = bool(diff) and max(diff) < 0.5 + 2 * BOUNDARY  # differing spans all scored < 0.7
        if label is not None:  # list of gold spans
            golds = [g['text'] for g in label]
            for who, a in (('ref', r), ('cand', c)):
                pred = top(a)
                row[f'{who}_gold_f1'] = (float(pred == '') if not golds else max(f1(pred, g) for g in golds))
                row[f'{who}_gold_em'] = (pred == '' if not golds else any(squad_norm(pred) == squad_norm(g) for g in golds))
        return row
    pr, pc = probs(r), probs(c)
    row['kl'] = kl(pr, pc)
    row['max_abs_dp'] = max(abs(a - b) for a, b in zip(pr, pc))
    if t == 'noul':
        row.update(p_ref=r['noul'], p_cand=c['noul'], abs_dp=abs(r['noul'] - c['noul']),
                   dlogit=logit(c['noul']) - logit(r['noul']), agree=(r['noul'] >= 0.5) == (c['noul'] >= 0.5),
                   boundary=abs(r['noul'] - 0.5) < BOUNDARY)
        if label is not None:
            row['ref_correct'] = (r['noul'] >= 0.5) == bool(label)
            row['cand_correct'] = (c['noul'] >= 0.5) == bool(label)
    elif t == 'choice':
        rp, cp = r['probabilities'], c['probabilities']
        rtop, ctop = max(rp, key=rp.get), max(cp, key=cp.get)
        srt = sorted(rp.values(), reverse=True)
        margin = srt[0] - (srt[1] if len(srt) > 1 else 0)
        row.update(agree=rtop == ctop, abs_dp=row['max_abs_dp'], boundary=margin < 2 * BOUNDARY, ref_margin=margin)
        if label is not None:
            row['ref_correct'] = rtop == label
            row['cand_correct'] = ctop == label
    elif t == 'score':
        n = len(pr)
        rtop, ctop = pr.index(max(pr)), pc.index(max(pc))
        row.update(agree=rtop == ctop, abs_dscore=abs(r['score'] - c['score']), abs_dp=row['max_abs_dp'],
                   score_ref=r['score'], score_cand=c['score'], boundary=abs(r['score'] - round(r['score'])) > 0.5 - BOUNDARY * (n - 1) / 2)
        if label is not None:  # BoolQ: yes if score above the midpoint
            mid = (n - 1) / 2
            row['ref_correct'] = (r['score'] > mid) == bool(label)
            row['cand_correct'] = (c['score'] > mid) == bool(label)
    return row


def wilson(k, n, z=1.96):
    if n == 0:
        return [0, 0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summarize(rows):
    out = {}
    by_type = {}
    for r in rows:
        by_type.setdefault(r['type'], []).append(r)
    for t, rs in by_type.items():
        n = len(rs)
        s = dict(n=n)
        if t == 'locate':
            for k in ('span_exact', 'empty_agree', 'top_text_equal'):
                s[k] = sum(r[k] for r in rs)
            s['mean_char_jaccard'] = sum(r['char_jaccard'] for r in rs) / n
            s['max_shared_span_dscore'] = max(r['shared_span_max_dscore'] for r in rs)
            s['ref_only_spans'] = sum(r['ref_only_spans'] for r in rs)
            s['cand_only_spans'] = sum(r['cand_only_spans'] for r in rs)
            s['boundary_span_flips'] = sum(r['boundary_span_flip'] for r in rs)
            lab = [r for r in rs if 'ref_gold_f1' in r]
            if lab:
                s['labeled_n'] = len(lab)
                for who in ('ref', 'cand'):
                    s[f'{who}_gold_f1'] = sum(r[f'{who}_gold_f1'] for r in lab) / len(lab)
                    s[f'{who}_gold_em'] = sum(r[f'{who}_gold_em'] for r in lab)
        else:
            agree = sum(r['agree'] for r in rs)
            s.update(agree=agree, agree_ci95=wilson(agree, n), flips=n - agree,
                     mean_abs_dp=sum(r['abs_dp'] for r in rs) / n, max_abs_dp=max(r['abs_dp'] for r in rs),
                     mean_kl=sum(r['kl'] for r in rs) / n, max_kl=max(r['kl'] for r in rs))
            b = [r for r in rs if r['boundary']]
            s.update(boundary_n=len(b), boundary_flips=sum(not r['agree'] for r in b),
                     nonboundary_flips=sum(not r['agree'] for r in rs if not r['boundary']))
            if t == 'noul':
                d = [r['dlogit'] for r in rs]
                mu = sum(d) / n
                sd = math.sqrt(sum((x - mu) ** 2 for x in d) / max(1, n - 1))
                s.update(mean_dlogit=mu, sd_dlogit=sd, max_abs_dlogit=max(abs(x) for x in d))
                # flips expected if the candidate's error were N(mu, sd) noise on the reference logit
                s['expected_flips_noise'] = sum(phi(-logit(r['p_ref']) / sd) if logit(r['p_ref']) > 0 else phi(logit(r['p_ref']) / sd)
                                                for r in rs) if sd > 0 else 0.0
                s['flip_ids'] = [r['id'] for r in rs if not r['agree']]
            if t == 'score':
                s['mean_abs_dscore'] = sum(r['abs_dscore'] for r in rs) / n
                s['max_abs_dscore'] = max(r['abs_dscore'] for r in rs)
            lab = [r for r in rs if 'ref_correct' in r]
            if lab:
                s['labeled_n'] = len(lab)
                s['ref_label_acc'] = sum(r['ref_correct'] for r in lab)
                s['cand_label_acc'] = sum(r['cand_correct'] for r in lab)
        out[t] = s
    return out


def compare(ref, cand, sets, labels):
    rows, missing, errors = [], [], []
    for i, rans in ref.items():
        if '__error__' in rans:
            continue
        if i not in cand:
            missing.append(i)
            continue
        cans = cand[i]
        if '__error__' in cans:
            errors.append(dict(id=i, error=cans['__error__']))
            continue
        lab = labels.get(i)
        for k, r in rans.items():
            if k not in cans:
                errors.append(dict(id=i, error=f'missing answer {k}'))
                continue
            l = lab.get(k) if isinstance(lab, dict) else lab
            if r['type'] == 'locate' and not isinstance(l, list) or r['type'] != 'locate' and isinstance(l, list):
                l = None
            if r['type'] == 'choice' and isinstance(l, bool):
                l = None  # boolq choice: label mapped below
                if isinstance(lab, bool) and set(r['probabilities']) >= {'yes', 'no'}:
                    l = 'yes' if lab else 'no'
            row = compare_item(k, r, cans[k], l)
            row.update(id=i, set=sets.get(i, '?'))
            rows.append(row)
    return rows, missing, errors


def bootstrap_diff(rows_a, rows_b, metric, n=2000, seed=0):
    """paired bootstrap over items for mean(metric_a) - mean(metric_b)"""
    pa = {(r['id'], r['key']): r for r in rows_a if metric in r}
    pb = {(r['id'], r['key']): r for r in rows_b if metric in r}
    keys = sorted(set(pa) & set(pb))
    if not keys:
        return None
    d = [float(pa[k][metric]) - float(pb[k][metric]) for k in keys]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return dict(n=len(d), diff=sum(d) / len(d), ci95=[means[int(0.025 * n)], means[int(0.975 * n)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', nargs='+', required=True)
    ap.add_argument('--cand', required=True)
    ap.add_argument('--cand-b')
    ap.add_argument('--eval', nargs='*', help='evaluation JSONL files (labels, set names; default: DATA/eval/*.jsonl)')
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--subset', action='store_true',
                    help='compare only the ids the candidate answered (partial runs); otherwise they count as missing')
    ap.add_argument('--locate-min-score', type=float, default=0.0, help='ignore Locate spans below this score (use 0.5 for --locate-threshold runs)')
    ap.add_argument('--cand-inputs', nargs='*', default=[], help="the candidate's own input JSONL(s): verify its requests equal the eval requests")
    ap.add_argument('--out')
    ap.add_argument('--name', default='')
    ap.add_argument('--rows', action='store_true', help='include per-question rows in the report')
    a = ap.parse_args()
    if a.eval is None:
        a.eval = [str(p) for p in sorted((a.data / 'eval').glob('*.jsonl'))]
    global LOCATE_MIN
    LOCATE_MIN = a.locate_min_score
    ref, sets = {}, {}
    for spec in a.ref:
        o, s = load_outputs(spec)
        ref.update(o); sets.update(s)
    labels = {}
    for e in a.eval:
        for r in jl(e):
            labels[r['id']] = r.get('label', r.get('reference_label'))
            sets[r['id']] = r.get('set', Path(e).stem)
    cand, cs = load_outputs(a.cand)
    # request identity: eval requests vs what the candidate/reference actually ran
    ereq = {}
    for e in a.eval:
        for r in jl(e):
            ereq[r['id']] = hashlib.sha256(json.dumps(r['request'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    seen = {}
    for spec in [a.cand] + a.ref:
        seen.update(REQ_HASH.get(spec, {}))
    for f in a.cand_inputs:
        for r in jl(f):
            req = r.get('request', r)
            seen[r.get('id')] = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    bad = sorted(i for i, h in seen.items() if i in ereq and ereq[i] != h)
    for i in bad:
        cand[i] = {'__error__': 'request differs from the eval set (stale input?)'}
    if bad:
        print(f'WARNING: {len(bad)} ids ran on a different request than the eval set; counted as errors', file=sys.stderr)
    for k, v in cs.items():
        sets.setdefault(k, v)
    if a.subset:
        ref = {k: v for k, v in ref.items() if k in cand}
    rows, missing, errors = compare(ref, cand, sets, labels)
    report = dict(name=a.name, ref=a.ref, cand=a.cand, subset=a.subset, compared_questions=len(rows), missing=len(missing),
                  missing_ids=missing[:20], errors=errors[:20], n_errors=len(errors))
    by_set = {}
    for r in rows:
        by_set.setdefault(r['set'], []).append(r)
    report['sets'] = {k: summarize(v) for k, v in sorted(by_set.items())}
    report['all'] = summarize(rows)
    if a.cand_b:
        cb, _ = load_outputs(a.cand_b)
        rows_b, _, _ = compare(ref, cb, sets, labels)
        report['paired_vs_b'] = dict(cand_b=a.cand_b, **{m: bootstrap_diff(rows, rows_b, m) for m in ('abs_dp', 'kl', 'agree')})
    if a.rows:
        report['rows'] = rows
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text + '\n')
    print_table(report)


def print_table(rep):
    print(f"# {rep['name'] or rep['cand']}  questions={rep['compared_questions']} missing={rep['missing']} errors={rep['n_errors']}")
    for set_name, s in list(rep['sets'].items()) + [('ALL', rep['all'])]:
        for t, m in s.items():
            if t == 'locate':
                extra = f" gold_f1 ref/cand={m['ref_gold_f1']:.3f}/{m['cand_gold_f1']:.3f}" if 'ref_gold_f1' in m else ''
                print(f"{set_name:24s} {t:7s} n={m['n']:4d} span_exact={m['span_exact']} empty_agree={m['empty_agree']} top_text={m['top_text_equal']} jacc={m['mean_char_jaccard']:.4f} ref_only/cand_only={m['ref_only_spans']}/{m['cand_only_spans']} boundary(<0.7)={m['boundary_span_flips']}{extra}")
            else:
                extra = f" dscore={m['mean_abs_dscore']:.4f}" if t == 'score' else ''
                extra += f" sd_dlogit={m['sd_dlogit']:.3f} E[flips|noise]={m['expected_flips_noise']:.2f}" if t == 'noul' else ''
                lab = f" label ref/cand={m['ref_label_acc']}/{m['cand_label_acc']}" if 'ref_label_acc' in m else ''
                print(f"{set_name:24s} {t:7s} n={m['n']:4d} agree={m['agree']} flips={m['flips']} (boundary {m['boundary_flips']}/{m['boundary_n']}) "
                      f"mean|dp|={m['mean_abs_dp']:.4f} max|dp|={m['max_abs_dp']:.4f} KL={m['mean_kl']:.5f}{extra}{lab}")


if __name__ == '__main__':
    main()

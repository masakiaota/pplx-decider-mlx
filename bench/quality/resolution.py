"""Resolution of flip counts: what flip counts does pure logit noise of size sigma produce on a reference set?

For each noul question with reference logit l, a candidate whose logit error is N(0, sigma^2) flips with
probability q = Phi(-|l|/sigma).  Flip count is Poisson-binomial(q_i).  We report its mean, sd, a 90% range,
and P(flips >= 5) (= failing the 96/100 gate on boolq100).

    python bench/quality/resolution.py REF.jsonl [REF2.jsonl ...] [--sigma 0.1 0.25 ...] [--out x.json]
"""
import argparse, json, math
from pathlib import Path


def logit(p, eps=1e-7):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def poisson_binomial(qs):
    dist = [1.0]
    for q in qs:
        new = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            new[k] += v * (1 - q)
            new[k + 1] += v * q
        dist = new
    return dist


def analyse(logits, sigma):
    qs = [phi(-abs(l) / sigma) for l in logits]
    d = poisson_binomial(qs)
    cdf, lo, hi = 0, None, None
    for k, v in enumerate(d):
        cdf += v
        if lo is None and cdf >= 0.05:
            lo = k
        if hi is None and cdf >= 0.95:
            hi = k
    mean = sum(qs)
    return dict(sigma=sigma, n=len(logits), mean_flips=mean, sd_flips=math.sqrt(sum(q * (1 - q) for q in qs)),
                range90=[lo, hi], p_ge5=sum(d[5:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('refs', nargs='+')
    ap.add_argument('--sigma', nargs='*', type=float, default=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5])
    ap.add_argument('--out')
    a = ap.parse_args()
    out = {}
    for ref in a.refs:
        logits = []
        for line in Path(ref).read_text().splitlines():
            r = json.loads(line)
            if 'error' in r or not isinstance(r.get('response'), dict):
                continue
            for ans in r['response']['answers'].values():
                if ans['type'] == 'noul':
                    logits.append(logit(ans['noul']))
        rows = [analyse(logits, s) for s in a.sigma]
        out[ref] = dict(n=len(logits), near_boundary_01=sum(abs(1 / (1 + math.exp(-l)) - 0.5) < 0.1 for l in logits), table=rows)
        print(f'# {ref} n={len(logits)} |p-0.5|<0.1: {out[ref]["near_boundary_01"]}')
        for r in rows:
            print(f"  sigma={r['sigma']:.2f} E[flips]={r['mean_flips']:.2f} sd={r['sd_flips']:.2f} 90%={r['range90']} P(>=5)={r['p_ge5']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

"""In-process timings for one model: load, preparation, single requests, a 20-request array, long inputs, bulk.

    uv run python bench/speed.py --model pplx-decider-flash-v1.3 [--long 5400,12900,32400] [--bulk 256] [--json out.json]

The inputs are synthetic (no dataset needed). Each case runs once to trace, then `--repeat` times; the median is
printed. Run nothing else on the GPU meanwhile. The numbers in README.md were measured over HTTP with the same
shapes on a Mac Studio (M5 Ultra); in-process times are a little lower.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import decider_mlx  # noqa: F401  (before mlx)

WORDS = ('alpha river stone window garden metal paper cloud orange signal market valley engine forest silver number '
         'bridge planet coffee summer').split()


def text(tokens, seed=0):
    n = int(tokens * 0.78)      # about 1.3 words per token in this tokenizer family
    return ' '.join(WORDS[(i * 7 + seed) % len(WORDS)] + ('.' if i % 13 == 12 else '') for i in range(n))


def request(tokens, seed=0, kind='noul'):
    q = {'noul': {'type': 'noul', 'instructions': 'Does the text mention a river?'},
         'choice': {'type': 'choice', 'instructions': 'Which word comes first?', 'criteria': {'a': 'alpha', 'b': 'river'}},
         'score': {'type': 'score', 'instructions': 'How often is a river mentioned?', 'criteria': ['never', 'once', 'often']}}[kind]
    return {'state': text(tokens, seed), 'questions': {'q': q}}


def timed(fn, repeat):
    fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter(); fn(); times.append(time.perf_counter() - t)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--repeat', type=int, default=5)
    ap.add_argument('--long', default='5400,12900,32400', help='token counts of the long single requests ("" to skip)')
    ap.add_argument('--bulk', type=int, default=256, help='requests in the bulk array (0 to skip)')
    ap.add_argument('--json', help='write the results here')
    a = ap.parse_args()
    out = {'model': a.model}
    t = time.perf_counter()
    app = decider_mlx.load(a.model)
    out['load_s'] = time.perf_counter() - t
    app.wait_ready()
    out['ready_s'] = time.perf_counter() - t
    print(f"{a.model}: load() {out['load_s']:.2f} s, preparation done {out['ready_s']:.1f} s")

    cases = [('single 427 tokens', lambda: app.decide(request(427))),
             ('single 427 tokens, 3 question types', lambda: app.decide(
                 {'state': text(427), 'questions': {k: request(0, 0, k)['questions']['q'] for k in ('noul', 'choice', 'score')}})),
             ('array of 20 (150-320 tokens)', lambda: app.decide([request(150 + 9 * i, i) for i in range(20)])),
             ('array of 20, isolated', lambda: app.decide([request(150 + 9 * i, i) for i in range(20)], execution_mode='isolated'))]
    for n in [int(x) for x in a.long.split(',') if x]:
        cases.append((f'long {n} tokens', lambda n=n: app.decide(request(n))))
    for name, fn in cases:
        s = timed(fn, a.repeat if 'long' not in name else max(1, a.repeat // 2))
        out[name] = s
        print(f'{name:40s} {s * 1000:9.1f} ms')
    if a.bulk:
        reqs = [request(100 + (i * 37) % 900, i, ('noul', 'choice', 'score')[i % 3]) for i in range(a.bulk)]
        s = timed(lambda: app.decide(reqs), 1)
        out[f'bulk {a.bulk}'] = s
        print(f'{f"bulk array of {a.bulk} (100-1000 tokens)":40s} {s * 1000:9.1f} ms  ({s / a.bulk * 1000:.1f} ms per request)')
    out['peak_bytes'] = app.metrics()['peak_bytes']
    print(f"MLX peak {out['peak_bytes'] / 1e9:.2f} GB")
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()

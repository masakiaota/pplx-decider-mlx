"""Bit-exactness against the golden outputs (tests/golden): boolq100 in 4 conditions and every HTTP feature case.
Golden lines are compared byte for byte (sorted keys, compact separators)."""
import base64
import http.client
import json
import threading
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / 'golden'
DIR = {'pplx-decider-flash-v1.3': 'flash', 'pplx-decider-v1.3': '27b'}
OTHER = {'pplx-decider-flash-v1.3': 'pplx-decider-v1.3', 'pplx-decider-v1.3': 'pplx-decider-flash-v1.3'}


def dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def line(record, result):
    d = dict(id=record['id'], set=record.get('set'))
    if isinstance(result, Exception):
        d['error'] = f'{type(result).__name__}: {result}'
    else:
        d['response'] = result
    return dump(d)


@pytest.mark.parametrize('mode', ['separate', 'isolated'])
@pytest.mark.parametrize('shape', ['single', 'list'])
def test_boolq(app, model, mode, shape):
    boolq = read(GOLDEN / 'boolq100.jsonl')
    golden = (GOLDEN / DIR[model] / f'boolq-{mode}-{shape}.jsonl').read_text().splitlines()
    if shape == 'single':
        got = [line(r, app.decide(r['request'], execution_mode=mode, temperature=1.0)) for r in boolq]
    else:
        got = [line(r, v) for r, v in zip(boolq, app.decide([r['request'] for r in boolq], execution_mode=mode, temperature=1.0))]
    different = [r['id'] for r, a, b in zip(boolq, got, golden) if a != b]
    assert not different, f'{len(different)} differ: {different[:5]}'


def _cases(model):
    images = json.loads((GOLDEN / 'http-images.json').read_text())
    names = {'@MODEL': model, '@OTHER': OTHER[model], '@LATEST': model.replace('-v1.3', '-latest'), '@PREFIXED': 'perplexity/' + model}

    def expand(x):
        if isinstance(x, str):
            if x.startswith('@image:'):
                return images[int(x[7:])]
            if x.startswith('@repeat:'):
                text, count = x[8:].rsplit(':', 1)
                return text * int(count)
            return names.get(x, x)
        if isinstance(x, list):
            return [expand(y) for y in x]
        if isinstance(x, dict):
            return {k: expand(v) for k, v in x.items()}
        return x
    for case in read(GOLDEN / 'http-cases.jsonl'):
        payload = base64.b64decode(case['payload_b64']) if 'payload_b64' in case else expand(case.get('payload'))
        yield case['case'], case['method'], case['path'], payload


def _call(port, method, path, payload):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=900)
    if method == 'BIG':        # Content-Length over the limit, body not sent
        c.putrequest('POST', path); c.putheader('Content-Type', 'application/json')
        c.putheader('Content-Length', str(33 * 1024 * 1024)); c.endheaders()
    else:
        data = payload if isinstance(payload, bytes) or payload is None else json.dumps(payload).encode()
        c.request(method, path, body=data, headers={'Content-Type': 'application/json'})
    r = c.getresponse(); b = r.read(); c.close()
    try:
        b = json.loads(b) if b else None
    except ValueError:
        b = b.decode(errors='replace')
    return dict(status=r.status, body=b, headers={k: r.getheader(k) for k in ('Allow', 'Content-Type', 'Retry-After') if r.getheader(k) is not None})


def test_http(app, model):
    from decider_mlx.server.http import make_server
    server = make_server('127.0.0.1', 0, lambda: app)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        golden = {json.loads(g)['case']: g for g in (GOLDEN / DIR[model] / 'http.jsonl').read_text().splitlines()}
        different = []
        for name, method, path, payload in _cases(model):
            got = dump(dict(case=name, **_call(port, method, path, payload)))
            if got != golden[name]:
                different.append(name)
        assert not different, f'{len(different)} differ: {different[:10]}'
    finally:
        server.shutdown()

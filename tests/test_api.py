"""API behaviour that needs no model: model names, the HTTP body checks, per-item array errors, error wording, the
CLI. One test (test_app_threads) runs a model when DECIDER_MLX_TEST_MODELS is set."""
import json
import threading

import pytest

from decider_mlx import registry
from decider_mlx.protocol import LengthError, RequestError, validate
from decider_mlx.server import http

FLASH, LARGE = 'pplx-decider-flash-v1.3', 'pplx-decider-v1.3'
S = 'The museum opened in 1931.'
Q = {'a': {'type': 'noul', 'instructions': 'Did it open before 1950?'}}


class FakeApp:
    def __init__(self, model):
        self.spec = registry.spec(model); self.model = model; self.calls = []

    def _one(self, r):
        if r['state'] == 'too long':
            return LengthError("Question 'a' needs 40000 tokens")
        try:
            validate(r)
        except RequestError as e:
            return e
        return {'model': self.model, 'answers': {k: {'type': 'noul', 'noul': 0.5} for k in r['questions']}, 'usage': {}}

    def decide(self, request, **kw):
        self.calls.append(('decide', request, kw)); out = self._one(request)
        if isinstance(out, Exception):
            raise out
        return out

    def decide_each(self, requests, **kw):
        self.calls.append(('decide_each', requests, kw)); return [self._one(r) for r in requests]


def body(**kw):
    return {'model': FLASH, 'state': S, 'questions': Q} | kw


def test_registry_names():
    assert set(registry.MODELS) == {FLASH, LARGE}
    for name in ('pplx-decider-flash-latest', 'perplexity/pplx-decider-flash-v1.3', 'pplx-decider-flash-v1.2', ''):
        with pytest.raises(registry.ModelNameError):
            registry.spec(name)


@pytest.mark.parametrize('name', ['pplx-decider-flash-latest', 'perplexity/pplx-decider-flash-v1.3', LARGE, 'gpt-4'])
def test_http_model_must_match(name):
    with pytest.raises(http.ApiError) as e:
        http.to_request(body(model=name), registry.spec(FLASH))
    assert e.value.status == 400 and e.value.param == 'model'


def test_http_response_model_is_request_model():
    app = FakeApp(FLASH)
    out = http.handle(app, body(), {})
    assert out['model'] == FLASH
    out = http.handle(app, [body(), body()], {'raw': ['1']})
    assert all(o['model'] == FLASH and 'logits' in o for o in out)


def test_array_bad_item_is_per_item():
    app = FakeApp(FLASH)
    out = http.handle(app, [body(), {'model': FLASH, 'state': S}, body(state='too long'), body(model='pplx-decider-flash-latest'), body()], {})
    assert [('error' in o) for o in out] == [False, True, True, True, False]
    assert out[1]['error']['param'] == 'questions'
    assert out[2]['error']['type'] == 'invalid_request_error' and not out[2]['error']['message'].startswith('LengthError')
    assert out[3]['error']['param'] == 'model'
    assert app.calls[-1][0] == 'decide_each' and len(app.calls[-1][1]) == 3   # HTTP-invalid items never reach the engine


def test_array_empty_is_400():
    with pytest.raises(http.ApiError):
        http.handle(FakeApp(FLASH), [], {})


@pytest.mark.parametrize('name', [FLASH, LARGE])
def test_score_level_wording_is_shared(name):
    spec = registry.spec(name)
    model = name
    for levels, text in ((['only'], 'a score needs at least two ordered levels'), ([str(i) for i in range(11)], 'at most 10')):
        with pytest.raises(http.ApiError) as e:
            http.to_request(body(model=model, questions={'s': {'type': 'score', 'criteria': levels}}), spec)
        assert text in str(e.value) and e.value.param == 'questions.s.criteria'


def test_error_body_has_no_class_prefix():
    assert http.error_body(RequestError('Question IDs must be nonempty strings'))['error']['message'] == 'Question IDs must be nonempty strings'
    assert http.error_body(RuntimeError('x'))['error']['type'] == 'api_error'


def test_query_options():
    assert http.query_options({}) == ('separate', None, False)
    assert http.query_options({'mode': ['isolated'], 'temperature': ['0.5'], 'raw': ['1']}) == ('isolated', 0.5, True)
    for q in ({'mode': ['x']}, {'temperature': ['abc']}, {'temperature': ['0']}, {'foo': ['1']}):
        with pytest.raises(http.ApiError):
            http.query_options(q)


def test_validate_wording():
    with pytest.raises(RequestError, match="type must be 'choice', 'noul', 'score' or 'locate'"):
        validate({'state': S, 'questions': {'x': {'type': 'yesno'}}})
    with pytest.raises(RequestError, match='at least one question'):
        validate({'state': S, 'questions': {}})


def test_cli_rejects_aliases(capsys):
    from decider_mlx.cli import main
    with pytest.raises(SystemExit):
        main(['serve', '--model', 'pplx-decider-flash-latest', '--port', '1'])
    assert 'Unknown model' in capsys.readouterr().err


def test_dev_override_is_internal():
    assert registry._DEV_ENV.startswith('_')


def test_app_threads(app, model):
    """decide() from several threads is served in order by the one worker; wait_ready() returns True."""
    req = {'state': S, 'questions': Q}
    results = [None] * 6

    def go(i):
        results[i] = app.decide(req, temperature=1.0)
    threads = [threading.Thread(target=go, args=(i,)) for i in range(6)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len({json.dumps(r, sort_keys=True) for r in results}) == 1
    assert results[0]['model'] == model
    assert app.wait_ready(timeout=600)
    assert app.status()['ready']
    with pytest.raises(ValueError):
        app.decide(req, temperature=0)

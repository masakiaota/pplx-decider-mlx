"""HTTP server: POST /v1/decisions with the official Decisions API body (https://docs.perplexity.ai/api-reference/decisions-post).

Body: {"model", "state", "questions"[, "mode", "max_staleness"]}; unknown top-level fields are 400. "model" must be the
served model's official name, and the response "model" is that name. Images are OpenAI-style parts
{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}} anywhere in `state` (PNG/JPEG/WebP data URLs,
at most 8); they are taken out of `state` in reading order and placed before the request text (model card). The
rest of `state` is kept as given (a string is text, an object or array is read as JSON). mode=web_grounding is not
available locally (400).

Local extensions: a JSON array of bodies (a same-order array of responses; an invalid item gets {"error": ...} in its
place; 64 or more are packed internally), query parameters mode=separate|isolated (default separate),
temperature=<number> (default: the model's saved temperature, as the official API) and raw=1 (readout logits).
Errors: {"error": {"message", "type", "code", "param"}}: 400 invalid request (including LengthError), 404 unknown
path, 405 other methods (empty body), 413 over 32 MiB, 500 model failure, 503 with Retry-After: 1 while loading.
GET /metrics: status and memory.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..protocol import TYPES, LengthError, RequestError
from ..registry import MODELS

log = logging.getLogger(__name__)
FIELDS = {'model', 'state', 'questions', 'mode', 'max_staleness'}
MAX_BODY = 32 * 1024 * 1024
MAX_IMAGES = 8
MAX_SCORE_LEVELS = 10     # the API reference allows 1 to 10 levels; the model needs at least 2
MIME = ('image/png', 'image/jpeg', 'image/webp')


class ApiError(Exception):
    def __init__(self, message, status=400, type='invalid_request_error', param=None):
        super().__init__(message)
        self.status, self.type, self.param = status, type, param

    def body(self):
        return {'error': {'message': str(self), 'type': self.type, 'code': None, 'param': self.param}}


def error_body(e):
    if isinstance(e, ApiError):
        return e.body()
    if isinstance(e, (RequestError, LengthError)):
        return ApiError(str(e)).body()
    return ApiError(f'{type(e).__name__}: {e}', 500, 'api_error').body()


def _image_url(part):
    value = part.get('image_url')
    url = value.get('url') if isinstance(value, dict) else value
    if not isinstance(url, str) or not url.startswith('data:'):
        raise ApiError('image_url must be a base64 data URL', param='state')
    head, sep, data = url.partition(',')
    mime = head[5:].split(';')[0].lower()
    if not sep or ';base64' not in head.lower() or mime not in MIME:
        raise ApiError('Unsupported image data URL: PNG, JPEG or WebP base64 data URLs only', param='state')
    try:
        base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ApiError('Image data URL is not valid base64', param='state')
    return f'data:{mime};base64,{data}'


_DROP = object()


def _extract(value, images):
    if isinstance(value, dict) and value.get('type') == 'image_url':
        images.append(_image_url(value))
        return _DROP
    if isinstance(value, list):
        return [v for v in (_extract(v, images) for v in value) if v is not _DROP]
    if isinstance(value, dict):
        return {k: v for k, v in ((k, _extract(v, images)) for k, v in value.items()) if v is not _DROP}
    return value


def model_error(model, served):
    if model in MODELS:
        return f'Model {model!r} is not served by this server; it serves {served!r}'
    return f'Unknown model {model!r}; this server serves {served!r}'


def to_request(body, spec):
    """Official body -> local request. Raises ApiError."""
    if not isinstance(body, dict):
        raise ApiError('Request body must be a JSON object')
    unknown = sorted(set(body) - FIELDS)
    if unknown:
        raise ApiError(f'Unknown field: {unknown[0]}', param=unknown[0])
    model = body.get('model')
    if not isinstance(model, str) or not model:
        raise ApiError('model is required', param='model')
    if model != spec.name:
        raise ApiError(model_error(model, spec.name), param='model')
    if 'mode' in body:
        raise ApiError('mode web_grounding is not available on this local server' if body['mode'] == 'web_grounding'
                       else "mode must be 'web_grounding'", param='mode')
    if 'state' not in body:
        raise ApiError('state is required', param='state')
    if not isinstance(body['state'], (str, dict, list)):
        raise ApiError('state must be a string, object or array', param='state')
    questions = body.get('questions')
    if not isinstance(questions, dict) or not questions:
        raise ApiError('questions must be a nonempty object', param='questions')
    for key, q in questions.items():
        if not key or not isinstance(q, dict):
            raise ApiError('Invalid question', param=f'questions.{key}')
        kind = q.get('type')
        if kind not in TYPES:
            raise ApiError(f"Question {key!r}: type must be 'choice', 'noul', 'score' or 'locate'", param=f'questions.{key}.type')
        if kind == 'noul' and q.get('instructions') is None and q.get('criteria') is None:
            raise ApiError('noul needs instructions or criteria', param=f'questions.{key}')
        if kind == 'score' and isinstance(q.get('criteria'), list) and len(q['criteria']) > MAX_SCORE_LEVELS:
            raise ApiError(f"Question {key!r}: {len(q['criteria'])} score levels; at most {MAX_SCORE_LEVELS} are supported",
                           param=f'questions.{key}.criteria')
        if kind == 'score' and isinstance(q.get('criteria'), list) and len(q['criteria']) < 2:
            raise ApiError(f'Question {key!r}: a score needs at least two ordered levels', param=f'questions.{key}.criteria')
        if kind == 'locate':
            ins = q.get('instructions')
            if not isinstance(ins, str) or not ins.strip():
                raise ApiError('locate instructions must be a nonempty string', param=f'questions.{key}.instructions')
            questions = {**questions, key: {'type': 'locate', 'instructions': ins}}
    images = []
    state = _extract(body['state'], images)
    if len(images) > MAX_IMAGES:
        raise ApiError(f'At most {MAX_IMAGES} images per question', param='state')
    if spec.join_image_text_state and images and isinstance(state, list):
        # 27B: [image, ..., "text" | {"type": "text", "text"}] reads like images=[...] with the joined text as state
        texts = [p if isinstance(p, str) else p.get('text') if isinstance(p, dict) and p.get('type') == 'text' and set(p) == {'type', 'text'} and isinstance(p.get('text'), str) else None for p in state]
        if all(t is not None for t in texts):
            state = '\n'.join(texts)
    request = {'state': state, 'questions': questions}
    if images:
        request['images'] = images
    return request


def query_options(q):
    unknown = sorted(set(q) - {'mode', 'temperature', 'raw'})
    if unknown:
        raise ApiError(f'Unknown query parameter: {unknown[0]}', param=unknown[0])
    mode = q.get('mode', ['separate'])[0]
    if mode not in ('separate', 'isolated'):
        raise ApiError('mode must be separate or isolated', param='mode')
    t = q.get('temperature', ['default'])[0]
    try:
        temperature = None if t == 'default' else float(t)
    except ValueError:
        raise ApiError('temperature must be a number', param='temperature')
    if temperature is not None and not (math.isfinite(temperature) and temperature > 0):
        raise ApiError('temperature must be finite and positive', param='temperature')
    return mode, temperature, q.get('raw', ['0'])[0] == '1'


def handle(app, body, query):
    """One POST body (object or array) -> the JSON response value. Raises ApiError / RequestError / LengthError."""
    mode, temperature, raw = query_options(query)
    spec, name = app.spec, app.model
    wrap = (lambda o: {'model': name, 'logits': o}) if raw else (lambda o: {**o, 'model': name})
    if not isinstance(body, list):
        return wrap(app.decide(to_request(body, spec), execution_mode=mode, temperature=temperature, return_logits=raw))
    if not body:
        raise ApiError('Empty list')
    out, valid = [None] * len(body), []
    for i, item in enumerate(body):
        try:
            valid.append((i, to_request(item, spec)))
        except ApiError as e:
            out[i] = e.body()
    if valid:
        results = app.decide_each([r for _, r in valid], execution_mode=mode, temperature=temperature, return_logits=raw)
        for (i, _), res in zip(valid, results):
            out[i] = error_body(res) if isinstance(res, Exception) else wrap(res)
    return out


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def make_server(host, port, get_app):
    """The HTTP server; get_app() returns the App, or None while loading (503)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, status=200, headers=()):
            data = b'' if value is None else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            if value is not None:
                self.send_header('Content-Type', 'application/json')
            if status == 405:
                self.send_header('Allow', 'POST')
            for k, v in headers:
                self.send_header(k, v)
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        def loading(self):
            self.send({'error': {'message': 'Model is loading; retry shortly', 'type': 'service_unavailable', 'code': None, 'param': None}},
                      503, [('Retry-After', '1')])

        def do_GET(self):
            if self.path == '/metrics':
                app = get_app()
                return self.loading() if app is None else self.send(app.metrics())
            self.send(None, 405 if self.path == '/v1/decisions' else 404)

        def do_PUT(self):
            self.send(None, 405 if self.path == '/v1/decisions' else 404)
        do_DELETE = do_PATCH = do_PUT

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != '/v1/decisions':
                return self.send(None, 404)
            app = get_app()
            if app is None:
                return self.loading()
            try:
                length = self.headers.get('Content-Length')
                if length is None:
                    raise ApiError('Content-Length is required')
                if int(length) > MAX_BODY:
                    return self.send(ApiError('Request body over 32 MiB', 413).body(), 413)
                try:
                    body = json.loads(self.rfile.read(int(length)))
                except ValueError:
                    raise ApiError('Invalid JSON')
                self.send(handle(app, body, parse_qs(u.query, keep_blank_values=True)))
            except Exception as e:   # noqa: BLE001
                body = error_body(e)
                status = getattr(e, 'status', 400 if isinstance(e, (RequestError, LengthError)) else 500)
                if status == 500:
                    log.exception('request failed')
                self.send(body, status)

    return _Server((host, port), Handler)


def serve(model, port, host='127.0.0.1', on_progress=None, on_listening=None):
    """Listen at once (503 until the weights are loaded), load `model`, then serve until interrupted."""
    from ..app import App
    holder = {}
    server = make_server(host, port, lambda: holder.get('app'))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if on_listening:
        on_listening(server)
    try:
        app = App(model)
        if on_progress:
            app.on_progress(on_progress)
            on_progress(app)
        holder['app'] = app
        thread.join()
    finally:
        server.shutdown()

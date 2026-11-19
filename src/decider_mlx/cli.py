"""Command line: decider-mlx serve | run | fetch.

  decider-mlx serve --model M --port P
  decider-mlx run --model M --input requests.jsonl --output responses.jsonl
  decider-mlx fetch --model M

M is pplx-decider-flash-v1.3 or pplx-decider-v1.3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .registry import MODELS, ModelNameError, spec


class Status:
    """One status line on stderr (rewritten in place on a terminal)."""

    def __init__(self, stream=sys.stderr):
        self.stream, self.tty, self.last = stream, stream.isatty(), None
        self.t0 = time.perf_counter()

    def __call__(self, text, final=False):
        if text == self.last and not final:
            return
        self.last = text
        line = f'[{time.perf_counter() - self.t0:6.1f}s] {text}'
        if self.tty:
            self.stream.write('\r\033[K' + line + ('\n' if final else ''))
        else:
            self.stream.write(line + '\n')
        self.stream.flush()


def _progress(status, accepting):
    def show(app):
        if app.ready:
            status(f'{app.model}: ready ({"accepting requests" if accepting else "running"}; preparation done in {app.prep_s:.1f} s)', final=True)
        else:
            status(f'{app.model}: preparing {app.prep_done}/{app.prep_total} ({"accepting requests" if accepting else "running"})')
    return show


def _model(value):
    try:
        return spec(value).name
    except ModelNameError as e:
        raise argparse.ArgumentTypeError(str(e))


def _ensure(model):
    """Download the weights first when they are not in the Hugging Face cache (with the download screen)."""
    from .fetch import Screen, snapshot
    snapshot(model, screen=Screen())


def cmd_serve(a):
    from .server.http import serve
    _ensure(a.model)
    status = Status()
    status(f'{a.model}: loading weights')
    serve(a.model, a.port, on_progress=_progress(status, True),
          on_listening=lambda server: status(f'{a.model}: listening on http://127.0.0.1:{a.port}/v1/decisions (503 until loaded)'))
    return 0


def cmd_run(a):
    from . import load
    lines = [json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    records = []
    for i, line in enumerate(lines):
        wrapped = isinstance(line, dict) and 'request' in line and 'questions' not in line
        o = (line.get('options') or {}) if wrapped else {}
        records.append(dict(id=line.get('id', i) if wrapped else i, request=line['request'] if wrapped else line,
                            mode=o.get('execution_mode', 'separate'), temperature=o.get('temperature'), raw=bool(o.get('raw', False))))
    _ensure(a.model)
    status = Status()
    status(f'{a.model}: loading weights')
    app = load(a.model)
    status(f'{a.model}: running {len(records)} requests')
    t = time.perf_counter()
    try:
        results = app.decide_records([{k: r[k] for k in ('request', 'mode', 'temperature', 'raw')} for r in records])
    except ValueError as e:      # invalid options
        print(f'decider-mlx: {e}', file=sys.stderr)
        return 2
    errors = 0
    with (a.output.open('w') if a.output else sys.stdout) as sink:
        for i, (r, res) in enumerate(zip(records, results)):
            line = dict(id=r['id'], index=i)
            if isinstance(res, Exception):
                line['error'] = dict(type=type(res).__name__, message=str(res)); errors += 1
            else:
                line['response'] = res
            sink.write(json.dumps(line, ensure_ascii=False) + '\n')
    status(f'{a.model}: {len(records)} requests ({errors} errors) in {time.perf_counter() - t:.1f} s', final=True)
    return 1 if errors else 0


def cmd_fetch(a):
    from .fetch import Screen, fetch
    print(fetch(a.model, screen=Screen()))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog='decider-mlx', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    names = ', '.join(MODELS)
    s = sub.add_parser('serve', help='HTTP server: POST /v1/decisions (official Decisions API body)')
    s.add_argument('--model', required=True, type=_model, help=names)
    s.add_argument('--port', required=True, type=int)
    s.set_defaults(fn=cmd_serve)
    r = sub.add_parser('run', help='JSONL file in, JSONL file out (one request per line, input order)')
    r.add_argument('--model', required=True, type=_model, help=names)
    from pathlib import Path
    r.add_argument('--input', required=True, type=Path,
                   help='one request per line, or {"id", "request", "options": {"execution_mode", "temperature", "raw"}}')
    r.add_argument('--output', type=Path, help='output JSONL: {"id", "index", "response" | "error"} per line (default stdout)')
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser('fetch', help='download the converted weights from Hugging Face (pinned revision) and check them')
    f.add_argument('--model', required=True, type=_model, help=names)
    f.set_defaults(fn=cmd_fetch)
    a = p.parse_args(argv)
    from .fetch import FetchError
    try:
        return a.fn(a)
    except FetchError as e:
        print(f'decider-mlx: {e}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())

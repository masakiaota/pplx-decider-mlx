"""A local stand-in for the Hugging Face Hub, enough for huggingface_hub's snapshot_download / hf_hub_download.

It serves local folders as model repositories at one commit each. The test suite uses it with a tiny synthetic
repository (test_fetch.py); it can also serve a full upload folder for an end-to-end check:

    python tests/hub_mock.py --port 8765 --repo masakiaota/pplx-decider-flash-v1.3-mlx=/path/to/folder [--rate-mb 500]
    HF_ENDPOINT=http://127.0.0.1:8765 HF_HUB_CACHE=/tmp/cache decider-mlx fetch --model pplx-decider-flash-v1.3

Any revision name resolves to the folder's commit (sha1 of its file list), except a full 40-hex commit hash, which
resolves to itself as on the real Hub. Files of 10 MiB or more, and every
.safetensors file, are served as LFS files whose ETag is their sha256 (taken from manifest.json when present).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

LFS_MIN = 10 * 1024 * 1024


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


class Repo:
    def __init__(self, folder):
        self.folder = Path(folder)
        files = sorted(p for p in self.folder.rglob('*') if p.is_file())
        manifest = {}
        if (self.folder / 'manifest.json').exists():
            manifest = json.loads((self.folder / 'manifest.json').read_text()).get('files', {})
        self.files = {}
        for p in files:
            rel = p.relative_to(self.folder).as_posix()
            size = p.stat().st_size
            lfs = size >= LFS_MIN or rel.endswith('.safetensors')
            if lfs:
                sha = manifest[rel]['sha256'] if rel in manifest and manifest[rel]['size'] == size else _sha256(p)
                etag = sha
            else:
                data = p.read_bytes()
                sha = hashlib.sha256(data).hexdigest()
                etag = hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
            self.files[rel] = dict(path=p, size=size, lfs=lfs, sha256=sha, etag=etag)
        self.commit = hashlib.sha1(json.dumps({k: v['etag'] for k, v in self.files.items()}, sort_keys=True).encode()).hexdigest()

    def tree(self):
        out, dirs = [], set()
        for rel, f in self.files.items():
            parts = rel.split('/')
            for i in range(1, len(parts)):
                dirs.add('/'.join(parts[:i]))
            entry = {'type': 'file', 'oid': f['etag'] if not f['lfs'] else hashlib.sha1(f['sha256'].encode()).hexdigest(),
                     'size': f['size'], 'path': rel}
            if f['lfs']:
                entry['lfs'] = {'oid': f['sha256'], 'size': f['size'], 'pointerSize': 134}
            out.append(entry)
        out += [{'type': 'directory', 'oid': hashlib.sha1(d.encode()).hexdigest(), 'size': 0, 'path': d} for d in sorted(dirs)]
        return out


def make_server(repos, port=0, rate=None):
    """repos: {repo_id: folder}. rate: bytes per second per file download (None: unlimited)."""
    repos = {k: Repo(v) for k, v in repos.items()}
    stats = {'requests': []}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def _json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(data)

        def _missing(self, kind):
            data = b'{"error": "not found"}'
            self.send_response(404)
            self.send_header('X-Error-Code', kind)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(data)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            stats['requests'].append((self.command, path))
            m = re.fullmatch(r'/api/models/([^/]+/[^/]+)/revision/([^/]+)', path)
            if m:
                repo = repos.get(m[1])
                if repo is None:
                    return self._missing('RepoNotFound')
                return self._json({'id': m[1], 'sha': commit_of(repo, m[2]), 'private': False,
                                   'siblings': [{'rfilename': f} for f in repo.files]})
            m = re.fullmatch(r'/api/models/([^/]+/[^/]+)/tree/([^/]+)(/.*)?', path)
            if m:
                repo = repos.get(m[1])
                if repo is None:
                    return self._missing('RepoNotFound')
                return self._json(repo.tree())
            m = re.fullmatch(r'/([^/]+/[^/]+)/resolve/([^/]+)/(.+)', path)
            if m:
                repo = repos.get(m[1])
                if repo is None:
                    return self._missing('RepoNotFound')
                f = repo.files.get(m[3])
                if f is None:
                    return self._missing('EntryNotFound')
                return self._file(repo, f, commit_of(repo, m[2]))
            self._missing('EntryNotFound')

        def _file(self, repo, f, commit):
            start, end = 0, f['size'] - 1
            rng = self.headers.get('Range')
            m = re.fullmatch(r'bytes=(\d+)-(\d*)', rng or '')
            self.send_response(206 if m and self.command == 'GET' else 200)
            if m and self.command == 'GET':
                start = int(m[1]); end = int(m[2]) if m[2] else end
                self.send_header('Content-Range', f'bytes {start}-{end}/{f["size"]}')
            self.send_header('X-Repo-Commit', commit)
            self.send_header('ETag', f'"{f["etag"]}"')
            if f['lfs']:
                self.send_header('X-Linked-Etag', f'"{f["etag"]}"')
                self.send_header('X-Linked-Size', str(f['size']))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            self.end_headers()
            if self.command == 'HEAD':
                return
            t0, sent = time.monotonic(), 0
            with open(f['path'], 'rb') as src:
                src.seek(start)
                left = end - start + 1
                while left > 0:
                    block = src.read(min(left, 1 << 22))
                    if not block:
                        break
                    self.wfile.write(block)
                    left -= len(block); sent += len(block)
                    if rate:
                        wait = sent / rate - (time.monotonic() - t0)
                        if wait > 0:
                            time.sleep(wait)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.repos, server.stats = repos, stats
    return server


def commit_of(repo, revision):
    """The commit a revision name resolves to: a full commit hash is itself, any other name the folder's commit."""
    return revision if re.fullmatch('[0-9a-f]{40}', revision) else repo.commit


def start(repos, rate=None):
    """Start a mock hub in a background thread; returns (server, endpoint URL)."""
    server = make_server(repos, rate=rate)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_address[1]}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--repo', action='append', required=True, help='REPO_ID=FOLDER (repeatable)')
    ap.add_argument('--rate-mb', type=float, help='limit each file download to this many MB/s')
    a = ap.parse_args()
    server = make_server(dict(r.split('=', 1) for r in a.repo), a.port, a.rate_mb * 1e6 if a.rate_mb else None)
    for name, repo in server.repos.items():
        print(f'{name}: {len(repo.files)} files, commit {repo.commit}', flush=True)
    print(f'listening on http://127.0.0.1:{a.port}', flush=True)
    server.serve_forever()

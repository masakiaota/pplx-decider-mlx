"""Download the converted weights from Hugging Face at the pinned revision (`decider-mlx fetch`).

Each model has one Hugging Face repository and one pinned revision in registry.py. The repository holds the weights,
the tokenizer, the processor and the configs, and `manifest.json` with the size and sha256 of every file. The files
go to the standard Hugging Face cache (~/.cache/huggingface, or HF_HUB_CACHE / HF_HOME).

- `snapshot(model)`: the local snapshot folder. It checks that every manifest file is present with its size and
  downloads the snapshot when not (serve, run and load() call it).
- `fetch(model)`: download what is missing, then check every file's sha256 against the manifest.

The download shows progress, speed and remaining time on stderr, with usage tips (tips.py), when `screen` is given
(the CLI); otherwise it only logs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

from .registry import spec

log = logging.getLogger(__name__)
MANIFEST = 'manifest.json'
EXTRA = ('README.md',)          # downloaded with the snapshot, not checked (the model card may be edited on the Hub)


class FetchError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------------- local

def _local_manifest(model):
    """Path of manifest.json in the cached snapshot of the pinned revision, or None."""
    from huggingface_hub import hf_hub_download
    s = spec(model)
    try:
        return Path(hf_hub_download(s.hf_repo, MANIFEST, revision=s.hf_revision, local_files_only=True))
    except Exception:   # noqa: BLE001 - any failure means "not in the cache"
        return None


def read_manifest(path):
    doc = json.loads(Path(path).read_text())
    if doc.get('format') != 'decider-mlx-manifest-v1' or not isinstance(doc.get('files'), dict):
        raise FetchError(f'{path}: not a decider-mlx manifest')
    return doc


def missing_files(root, manifest):
    """Manifest files that are absent or have the wrong size under root (fast: no hashing)."""
    out = []
    for name, entry in manifest['files'].items():
        p = Path(root) / name
        try:
            if p.stat().st_size != entry['size']:
                out.append(name)
        except OSError:
            out.append(name)
    return out


def snapshot(model, download=True, screen=None):
    """The local snapshot folder of `model`'s pinned revision. Downloads it (fetch) when files are missing and
    download is true; otherwise raises FetchError."""
    s = spec(model)
    path = _local_manifest(model)
    if path is not None and not missing_files(path.parent, read_manifest(path)):
        return path.parent
    if not download:
        raise FetchError(f'{model} is not downloaded; run: decider-mlx fetch --model {model}')
    log.info('%s: weights not found in the Hugging Face cache; downloading %s', model, s.hf_repo)
    return fetch(model, screen=screen)


# ------------------------------------------------------------------------------------------------------------- download

def _hub_error(model, e):
    s = spec(model)
    name = type(e).__name__
    if name in ('RepositoryNotFoundError', 'RevisionNotFoundError', 'GatedRepoError'):
        return FetchError(f'{model}: {s.hf_repo} at revision {s.hf_revision} is not available on Hugging Face ({name})')
    if name in ('LocalEntryNotFoundError', 'OfflineModeIsEnabled'):
        return FetchError(f'{model}: cannot reach Hugging Face and the weights are not in the local cache ({e})')
    return FetchError(f'{model}: download failed: {name}: {e}')


def _blob_bytes(folder):
    """Bytes on disk in a cache repository's blobs/ folder (complete and .incomplete files)."""
    total = 0
    try:
        with os.scandir(folder) as it:
            for entry in it:
                try:
                    total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _blobs_folder(model):
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name
    return Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id=spec(model).hf_repo, repo_type='model') / 'blobs'


def fetch(model, screen=None, verify=True):
    """Download `model` at its pinned revision (what is missing), check every file against manifest.json and return
    the snapshot folder."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    s = spec(model)
    try:
        manifest_path = Path(hf_hub_download(s.hf_repo, MANIFEST, revision=s.hf_revision))
    except Exception as e:   # noqa: BLE001
        raise _hub_error(model, e) from e
    manifest = read_manifest(manifest_path)
    if manifest.get('model') != model:
        raise FetchError(f'{s.hf_repo}: manifest.json is for {manifest.get("model")!r}, not {model!r}')
    root = manifest_path.parent
    total = sum(e['size'] for e in manifest['files'].values())
    todo = missing_files(root, manifest)
    if todo:
        blobs = _blobs_folder(model)
        known = {e['sha256'] for e in manifest['files'].values()}
        stale = sum(_blob_bytes_one(p) for p in _list(blobs) if p.name.split('.')[0] not in known and not p.name.endswith('.incomplete'))
        result, done = {}, threading.Event()

        def run():
            try:
                with disable_progress_bars():
                    result['path'] = snapshot_download(s.hf_repo, revision=s.hf_revision,
                                                       allow_patterns=sorted(manifest['files']) + [MANIFEST, *EXTRA])
            except BaseException as e:   # noqa: BLE001 - re-raised on the calling thread
                result['error'] = e
            finally:
                done.set()
        log.info('%s: downloading %d files (%.1f GB) from %s@%s', model, len(todo), total / 1e9, s.hf_repo, s.hf_revision)
        threading.Thread(target=run, name='decider-mlx-fetch', daemon=True).start()
        progress = Progress(total)
        progress.update(max(0, _blob_bytes(blobs) - stale))
        if screen is not None:
            screen.download(model, progress)          # the first tip at once
        last_log = time.monotonic()
        while not done.wait(0.25):
            progress.update(max(0, _blob_bytes(blobs) - stale))
            if screen is not None:
                screen.download(model, progress)
            elif time.monotonic() - last_log > 10:
                last_log = time.monotonic()
                log.info('%s: %s', model, progress.text())
        if 'error' in result:
            if screen is not None:
                screen.close()
            raise _hub_error(model, result['error']) from result['error']
        progress.update(total)
        if screen is not None:
            screen.download(model, progress, final=True)
        root = Path(result['path'])
        still = missing_files(root, manifest)
        if still:
            raise FetchError(f'{model}: {len(still)} files are missing after the download, for example {still[0]}')
    if verify:
        verify_files(model, root, manifest, screen)
    return root


def _list(folder):
    try:
        return list(Path(folder).iterdir())
    except OSError:
        return []


def _blob_bytes_one(p):
    try:
        return p.stat().st_size
    except OSError:
        return 0


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_files(model, root, manifest=None, screen=None):
    """Check every manifest file's size and sha256. Raises FetchError naming the files that differ."""
    manifest = manifest or read_manifest(Path(root) / MANIFEST)
    files = manifest['files']
    total = sum(e['size'] for e in files.values())
    bad = missing_files(root, manifest)
    done = 0
    progress = Progress(total)
    for i, (name, entry) in enumerate(sorted(files.items())):
        if name not in bad and sha256(Path(root) / name) != entry['sha256']:
            bad.append(name)
        done += entry['size']
        progress.update(done)
        if screen is not None:
            screen.verify(model, progress, i + 1, len(files), final=i + 1 == len(files))
    if bad:
        raise FetchError(f'{model}: {len(bad)} files do not match manifest.json ({", ".join(sorted(bad)[:3])}); '
                         f'delete them from the Hugging Face cache and run decider-mlx fetch --model {model} again')
    log.info('%s: %d files verified (sha256)', model, len(files))


# ------------------------------------------------------------------------------------------------------------- display

class Progress:
    """Bytes done of a total, with a smoothed rate and the remaining time."""

    def __init__(self, total):
        self.total, self.done, self.t0 = max(1, total), 0, time.monotonic()
        self._samples = [(self.t0, 0)]

    def update(self, done):
        now = time.monotonic()
        self.done = min(self.total, max(self.done, done))
        self._samples.append((now, self.done))
        while len(self._samples) > 2 and now - self._samples[0][0] > 10:
            self._samples.pop(0)

    @property
    def rate(self):
        (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
        return (b1 - b0) / (t1 - t0) if t1 > t0 else 0.0

    @property
    def eta(self):
        r = self.rate
        return (self.total - self.done) / r if r > 0 else None

    def text(self):
        eta = self.eta
        left = '--' if eta is None else _duration(eta)
        return (f'{self.done / 1e9:5.1f} / {self.total / 1e9:.1f} GB  {100 * self.done / self.total:3.0f}%  '
                f'{self.rate / 1e6:6.0f} MB/s  {left} left')

    def bar(self, width=30):
        n = int(width * self.done / self.total)
        return '[' + '#' * n + '.' * (width - n) + ']'


def _duration(s):
    s = int(round(s))
    return f'{s // 3600}h{s // 60 % 60:02d}m' if s >= 3600 else f'{s // 60}m{s % 60:02d}s' if s >= 60 else f'{s}s'


class Screen:
    """The download screen on stderr: one progress line and a tip that changes every `every` seconds. On a terminal
    the block is redrawn in place; otherwise a progress line is written every 10 s and each tip once."""

    def __init__(self, stream=None, every=8.0, width=100):
        self.stream = stream or sys.stderr
        self.tty = self.stream.isatty()
        self.every, self.width = every, width
        self.t0 = time.monotonic()
        self.lines = 0
        self.last_plain = 0.0
        self.tip_shown = -1

    def _tip_index(self):
        return int((time.monotonic() - self.t0) // self.every)

    def _block(self, head):
        from .tips import TIPS, tip
        i = self._tip_index()
        title, lines = tip(i)
        out = [head, '', f'  Tip {i % len(TIPS) + 1}/{len(TIPS)}: {title}']
        out += ['    ' + line for line in lines]
        return out, i

    def _draw(self, lines):
        if self.lines:
            self.stream.write(f'\033[{self.lines}F')
        for line in lines:
            self.stream.write('\033[2K' + line[:self.width + 20] + '\n')
        for _ in range(self.lines - len(lines)):
            self.stream.write('\033[2K\n')
        self.lines = max(self.lines, len(lines))
        self.stream.flush()

    def download(self, model, progress, final=False):
        head = f'Downloading {model}  {progress.bar()}  {progress.text()}'
        if final:
            head = f'Downloaded {model}: {progress.total / 1e9:.1f} GB in {_duration(time.monotonic() - progress.t0)}'
        if self.tty:
            block, _ = self._block(head)
            self._draw([head] if final else block)
            if final:
                self.lines = 0
            return
        now = time.monotonic()
        block, i = self._block(head)
        if i != self.tip_shown and not final:
            self.tip_shown = i
            self.stream.write('\n'.join(block[2:]) + '\n')
        if final or now - self.last_plain >= 10:
            self.last_plain = now
            self.stream.write(head + '\n')
        self.stream.flush()

    def verify(self, model, progress, i, n, final=False):
        line = f'Verifying {model}  {progress.bar()}  {i}/{n} files  {progress.done / 1e9:.1f} / {progress.total / 1e9:.1f} GB'
        if final:
            line = f'Verified {n} files of {model} (sha256)'
        if self.tty:
            self.stream.write('\r\033[K' + line + ('\n' if final else ''))
        elif final or time.monotonic() - self.last_plain >= 10:
            self.last_plain = time.monotonic()
            self.stream.write(line + '\n')
        self.stream.flush()

    def close(self):
        self.lines = 0

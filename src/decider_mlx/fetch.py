"""Download the converted weights from Hugging Face at the pinned revision (`decider-mlx fetch`).

Each model has one Hugging Face repository and one pinned revision in registry.py. The repository holds the weights,
the tokenizer, the processor and the configs, and `manifest.json` with the size and sha256 of every file. The files
go to the standard Hugging Face cache (~/.cache/huggingface, or HF_HUB_CACHE / HF_HOME).

- `snapshot(model)`: the local snapshot folder. It checks that every manifest file is present with its size and
  downloads the snapshot when not (serve, run and load() call it).
- `fetch(model)`: download what is missing using parallel HTTP Range requests (bypasses Xet overhead and splits large
  files across connections), verify every file's sha256 against the manifest, and return the snapshot folder.

The download shows progress, speed and remaining time on stderr, with usage tips (tips.py), when `screen` is given
(the CLI); otherwise it only logs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urljoin, urlsplit
import uuid

from .registry import spec

log = logging.getLogger(__name__)
MANIFEST = 'manifest.json'
EXTRA = ('README.md',)          # downloaded with the snapshot, not checked (the model card may be edited on the Hub)

MAX_WORKERS = 16                # concurrent HTTP streams (overridable via DECIDER_MLX_FETCH_WORKERS)
RANGE_CHUNK_SIZE = 16 << 20     # 16 MiB per Range request for large files
MULTIPART_MIN_SIZE = 16 << 20   # files larger than this are split across concurrent Range requests
STREAM_CHUNK_SIZE = 1 << 20     # 1 MiB read buffer per stream
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 120.0
ETAG_TIMEOUT = 60.0
MAX_RETRIES = 5


class FetchError(RuntimeError):
    pass


class _RetryableError(Exception):
    def __init__(self, message, delay=None, refresh_metadata=False):
        super().__init__(message)
        self.delay = delay
        self.refresh_metadata = refresh_metadata


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
    if isinstance(e, FetchError):
        return e
    s = spec(model)
    name = type(e).__name__
    if name in ('RepositoryNotFoundError', 'RevisionNotFoundError', 'GatedRepoError'):
        return FetchError(f'{model}: {s.hf_repo} at revision {s.hf_revision} is not available on Hugging Face ({name})')
    if name in ('LocalEntryNotFoundError', 'OfflineModeIsEnabled'):
        return FetchError(f'{model}: cannot reach Hugging Face and the weights are not in the local cache ({e})')
    return FetchError(f'{model}: download failed: {name}: {e}')


def _workers():
    raw = os.environ.get('DECIDER_MLX_FETCH_WORKERS')
    if raw and raw.isdigit():
        return max(1, min(64, int(raw)))
    return MAX_WORKERS


def _blobs_folder(root):
    """The repository's blobs/ folder corresponding to a snapshot folder `<repo>/snapshots/<commit>`."""
    return Path(root).parent.parent / 'blobs'


def _safe_etag(etag):
    """Normalize an HTTP ETag to the filename used in the Hugging Face blobs/ folder."""
    return etag.strip().strip('"').removeprefix('W/').strip('"')


def _existing_bytes(root, manifest):
    """Bytes of manifest files already present with the expected size in `root`."""
    total = 0
    for name, entry in manifest['files'].items():
        try:
            if (Path(root) / name).stat().st_size == entry['size']:
                total += entry['size']
        except OSError:
            pass
    return total


def _hub_origin(endpoint):
    parsed = urlsplit(endpoint)
    return (parsed.scheme, parsed.hostname, parsed.port)


@contextmanager
def _stream_url(client, url, headers, hub_origin, range_header=None):
    """Stream `url`, following redirects manually so Authorization is sent only to the Hub origin."""
    for _ in range(10):
        parsed = urlsplit(url)
        req_headers = {k: v for k, v in headers.items() if k.lower() != 'authorization'}
        if (parsed.scheme, parsed.hostname, parsed.port) == hub_origin:
            for k, v in headers.items():
                if k.lower() == 'authorization':
                    req_headers[k] = v
        req_headers['Accept-Encoding'] = 'identity'
        if range_header is not None:
            req_headers['Range'] = range_header
        with client.stream('GET', url, headers=req_headers, follow_redirects=False) as resp:
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get('location')
                if not location:
                    raise FetchError(f'Invalid redirect while downloading {url}')
                nxt = urljoin(url, location)
                if parsed.scheme == 'https' and urlsplit(nxt).scheme != 'https':
                    raise FetchError(f'Refused insecure redirect from HTTPS to {nxt}')
                url = nxt
                continue
            yield resp
            return
    raise FetchError(f'Too many redirects while downloading {url}')


def _retry_delay(attempt, resp=None):
    if resp is not None:
        ra = resp.headers.get('retry-after', '')
        if ra.isdigit():
            return min(int(ra), 120)
    return min(2 ** attempt, 30)


def _check_status(resp, filename):
    code = resp.status_code
    if code in (408, 429) or code >= 500:
        raise _RetryableError(f'{filename}: HTTP {code}', delay=_retry_delay(1, resp))
    if code in (401, 403):
        raise _RetryableError(f'{filename}: HTTP {code} (refreshing URL)', delay=1, refresh_metadata=True)
    if code not in (200, 206):
        raise FetchError(f'{filename}: unexpected HTTP {code}')


class _TransferCounter:
    """Thread-safe counter of unique downloaded bytes (retried bytes are never double-counted)."""

    def __init__(self, start=0):
        self._lock = threading.Lock()
        self._bytes = start

    def add(self, n):
        if n:
            with self._lock:
                self._bytes += n

    @property
    def value(self):
        with self._lock:
            return self._bytes


class _FileDownload:
    """State of one file being downloaded into the Hugging Face cache."""

    def __init__(self, repo_id, revision, filename, root, blobs, expected_size, expected_sha256, optional=False):
        self.repo_id = repo_id
        self.revision = revision
        self.filename = filename
        self.root = Path(root)
        self.blobs = Path(blobs)
        self.expected_size = expected_size      # None for EXTRA files
        self.expected_sha256 = expected_sha256  # None for EXTRA files
        self.optional = optional
        self.location = None
        self.etag = None
        self.size = expected_size
        self.blob_path = None
        self.tmp_path = None
        self.fd = None
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.remaining_parts = 0
        self.active_range_writers = 0
        self.range_bytes = 0
        self.full_fallback = False
        self.verified = False
        self.skipped = False                    # True when optional file is 404 or already cached

    @property
    def dst_path(self):
        return self.root / self.filename

    def fetch_metadata(self):
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
        url = hf_hub_url(self.repo_id, self.filename, revision=self.revision)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                info = get_hf_file_metadata(url, timeout=ETAG_TIMEOUT)
                break
            except EntryNotFoundError:
                if self.optional:
                    self.skipped = True
                    return False
                raise
            except HfHubHTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 404 and self.optional:
                    self.skipped = True
                    return False
                if (status in (408, 429) or (status is not None and status >= 500)) and attempt < MAX_RETRIES:
                    time.sleep(_retry_delay(attempt, e.response))
                    continue
                raise
            except Exception as e:
                import httpx
                if isinstance(e, httpx.TransportError) and attempt < MAX_RETRIES:
                    time.sleep(_retry_delay(attempt))
                    continue
                raise
        if not info.etag:
            raise FetchError(f'{self.filename}: missing ETag from Hub metadata')
        self.etag = _safe_etag(info.etag)
        self.location = info.location
        self.size = info.size if info.size is not None else self.expected_size
        if self.expected_size is not None and self.size is not None and self.size != self.expected_size:
            raise FetchError(f'{self.filename}: Hub size {self.size} differs from manifest {self.expected_size}')
        self.blob_path = self.blobs / self.etag
        return True

    def try_reuse_blob(self):
        """If the blob is already present with the right size (and sha256 when checked), link it and skip download."""
        try:
            if self.blob_path.is_file() and (self.size is None or self.blob_path.stat().st_size == self.size):
                if self.expected_sha256 is not None:
                    if sha256(self.blob_path) != self.expected_sha256:
                        return False
                    self.verified = True
                self._link_blob()
                self.skipped = True
                return True
        except OSError:
            pass
        return False

    def open_temp(self):
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.blobs / f'{self.etag}.{uuid.uuid4().hex[:8]}.incomplete'
        self.fd = os.open(self.tmp_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)

    def write_at(self, offset, data):
        view = memoryview(data)
        pos = 0
        while pos < len(view):
            written = os.pwrite(self.fd, view[pos:], offset + pos)
            if written <= 0:
                raise OSError(f'Short write to {self.tmp_path}')
            pos += written

    def close_fd(self):
        fd = self.fd
        if fd is not None:
            self.fd = None
            try:
                os.close(fd)
            except OSError:
                pass

    def cleanup(self):
        self.close_fd()
        if self.tmp_path is not None:
            try:
                self.tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def commit(self, computed_sha256=None):
        """Verify sha256 (if expected), atomically place the blob, and symlink into the snapshot."""
        self.close_fd()
        if self.expected_sha256 is not None:
            digest = computed_sha256 if computed_sha256 is not None else sha256(self.tmp_path)
            if digest != self.expected_sha256:
                self.cleanup()
                raise FetchError(f'{self.filename}: downloaded file does not match manifest.json sha256')
            self.verified = True
        os.replace(self.tmp_path, self.blob_path)
        self.tmp_path = None
        self._link_blob()

    def _link_blob(self):
        from huggingface_hub.file_download import _create_symlink
        self.dst_path.parent.mkdir(parents=True, exist_ok=True)
        _create_symlink(str(self.blob_path), str(self.dst_path), new_blob=True)


def _download_single(client, headers, hub_origin, state, counter, cancel):
    """Download a small (or unsized) file in one stream with resume-on-retry and streaming sha256."""
    import httpx
    if state.size == 0:
        state.open_temp()
        state.commit(hashlib.sha256().hexdigest())
        return
    state.open_temp()
    offset = 0
    digest = hashlib.sha256() if state.expected_sha256 is not None else None
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            if cancel.is_set():
                return
            rng = f'bytes={offset}-' if offset > 0 else None
            try:
                with _stream_url(client, state.location, headers, hub_origin, range_header=rng) as resp:
                    _check_status(resp, state.filename)
                    if offset > 0 and resp.status_code == 200:
                        # Server ignored Range on resume: restart from 0.
                        counter.add(-offset)
                        offset = 0
                        os.ftruncate(state.fd, 0)
                        if digest is not None:
                            digest = hashlib.sha256()
                    for block in resp.iter_bytes(chunk_size=STREAM_CHUNK_SIZE):
                        if cancel.is_set():
                            return
                        if not block:
                            continue
                        state.write_at(offset, block)
                        offset += len(block)
                        if digest is not None:
                            digest.update(block)
                        counter.add(len(block))
                if state.size is not None and offset != state.size:
                    raise _RetryableError(f'{state.filename}: truncated stream ({offset}/{state.size})', delay=1)
                state.commit(digest.hexdigest() if digest is not None else None)
                return
            except (_RetryableError, httpx.TransportError) as e:
                if cancel.is_set() or attempt >= MAX_RETRIES:
                    raise FetchError(f'{state.filename}: {e}') from e
                if isinstance(e, _RetryableError) and e.refresh_metadata:
                    state.fetch_metadata()
                delay = e.delay if isinstance(e, _RetryableError) and e.delay is not None else _retry_delay(attempt)
                if cancel.wait(delay):
                    return
    except BaseException:
        state.cleanup()
        raise


def _download_part(client, headers, hub_origin, state, start, end, counter, cancel):
    """Download one byte range [start, end] of a large file via os.pwrite, committing when the last part finishes."""
    import httpx
    expected = end - start + 1
    fallback_owner = False
    for attempt in range(1, MAX_RETRIES + 1):
        if cancel.is_set():
            return
        with state.lock:
            if state.full_fallback and not fallback_owner:
                return
        part_received = 0
        range_counted = 0
        try:
            with state.lock:
                url = state.location
            rng = None if fallback_owner else f'bytes={start}-{end}'
            with _stream_url(client, url, headers, hub_origin, range_header=rng) as resp:
                _check_status(resp, state.filename)
                if resp.status_code == 200:
                    # Server does not support Range requests: stop sibling writers and let one worker receive the body.
                    if not fallback_owner:
                        with state.cond:
                            if state.full_fallback:
                                return
                            state.full_fallback = True
                            state.cond.wait_for(lambda: state.active_range_writers == 0)
                            if state.range_bytes:
                                counter.add(-state.range_bytes)
                                state.range_bytes = 0
                        fallback_owner = True
                    os.ftruncate(state.fd, 0)
                    digest = hashlib.sha256() if state.expected_sha256 is not None else None
                    for block in resp.iter_bytes(chunk_size=STREAM_CHUNK_SIZE):
                        if cancel.is_set():
                            return
                        if not block:
                            continue
                        state.write_at(part_received, block)
                        part_received += len(block)
                        if digest is not None:
                            digest.update(block)
                        counter.add(len(block))
                    if state.size is not None and part_received != state.size:
                        raise _RetryableError(f'{state.filename}: truncated full response ({part_received}/{state.size})', delay=1)
                    state.commit(digest.hexdigest() if digest is not None else None)
                    return
                if resp.status_code != 206:
                    raise FetchError(f'{state.filename}: expected HTTP 206, got {resp.status_code}')
                with state.cond:
                    if state.full_fallback:
                        return
                    state.active_range_writers += 1
                try:
                    for block in resp.iter_bytes(chunk_size=STREAM_CHUNK_SIZE):
                        if cancel.is_set() or state.full_fallback:
                            return
                        if not block:
                            continue
                        if part_received + len(block) > expected:
                            raise FetchError(f'{state.filename}: range response exceeded expected length')
                        state.write_at(start + part_received, block)
                        part_received += len(block)
                        range_counted += len(block)
                        with state.lock:
                            state.range_bytes += len(block)
                        counter.add(len(block))
                finally:
                    with state.cond:
                        state.active_range_writers -= 1
                        state.cond.notify_all()
            if part_received != expected:
                raise _RetryableError(f'{state.filename}: short range ({part_received}/{expected})', delay=1)
            with state.lock:
                if state.full_fallback:
                    return
                state.remaining_parts -= 1
                done_file = state.remaining_parts == 0
            if done_file:
                state.commit()
            return
        except (_RetryableError, httpx.TransportError) as e:
            if range_counted:
                with state.lock:
                    drop = min(range_counted, state.range_bytes)
                    state.range_bytes -= drop
                if drop:
                    counter.add(-drop)
            elif part_received:
                counter.add(-part_received)
            if cancel.is_set() or attempt >= MAX_RETRIES:
                raise FetchError(f'{state.filename}: {e}') from e
            if isinstance(e, _RetryableError) and e.refresh_metadata:
                with state.lock:
                    state.fetch_metadata()
            delay = e.delay if isinstance(e, _RetryableError) and e.delay is not None else _retry_delay(attempt)
            if cancel.wait(delay):
                return


def _download_files(model, root, manifest, todo, extra_todo, counter, cancel):
    """Download all missing files in parallel using HTTP Range requests into the HF cache."""
    import httpx
    from huggingface_hub import constants
    from huggingface_hub.utils import build_hf_headers

    s = spec(model)
    blobs = _blobs_folder(root)
    blobs.mkdir(parents=True, exist_ok=True)
    revision = Path(root).name
    headers = build_hf_headers()
    hub_origin = _hub_origin(constants.ENDPOINT)
    workers = _workers()

    states = [
        _FileDownload(s.hf_repo, revision, name, root, blobs, manifest['files'][name]['size'], manifest['files'][name]['sha256'])
        for name in todo
    ]
    states += [
        _FileDownload(s.hf_repo, revision, name, root, blobs, None, None, optional=True)
        for name in extra_todo
    ]

    # Step 1: fetch metadata in parallel and reuse any matching cached blobs.
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(states)))) as meta_pool:
        futures = [meta_pool.submit(st.fetch_metadata) for st in states]
        for fut in as_completed(futures):
            fut.result()

    active = []
    for st in states:
        if st.skipped:
            continue
        if st.try_reuse_blob():
            if st.expected_size:
                counter.add(st.expected_size)
            continue
        active.append(st)

    if not active:
        return {st.filename for st in states if st.verified}

    # Step 2: schedule large files first (split into 16 MiB Range chunks) and small files on one shared pool.
    active.sort(key=lambda st: -(st.size or 0))
    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=30.0, pool=60.0)
    limits = httpx.Limits(max_connections=workers + 4, max_keepalive_connections=workers + 4)

    try:
        with httpx.Client(timeout=timeout, limits=limits, follow_redirects=False) as client:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='decider-mlx-dl') as pool:
                futures = []
                for st in active:
                    if st.size is not None and st.size > MULTIPART_MIN_SIZE:
                        parts = [(off, min(off + RANGE_CHUNK_SIZE, st.size) - 1) for off in range(0, st.size, RANGE_CHUNK_SIZE)]
                        st.open_temp()
                        st.remaining_parts = len(parts)
                        for start, end in parts:
                            futures.append(pool.submit(_download_part, client, headers, hub_origin, st, start, end, counter, cancel))
                    else:
                        futures.append(pool.submit(_download_single, client, headers, hub_origin, st, counter, cancel))
                try:
                    for fut in as_completed(futures):
                        fut.result()
                except BaseException:
                    cancel.set()
                    for fut in futures:
                        fut.cancel()
                    raise
    finally:
        for st in active:
            st.cleanup()

    return {st.filename for st in states if st.verified}


def fetch(model, screen=None, verify=True):
    """Download `model` at its pinned revision (what is missing), check every file against manifest.json and return
    the snapshot folder."""
    from huggingface_hub import hf_hub_download
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
    verified = set()
    if todo:
        extra_todo = [name for name in EXTRA if not (root / name).exists()]
        existing = _existing_bytes(root, manifest)
        counter = _TransferCounter(start=existing)
        cancel = threading.Event()
        result, done = {}, threading.Event()

        def run():
            try:
                result['verified'] = _download_files(model, root, manifest, todo, extra_todo, counter, cancel)
            except BaseException as e:   # noqa: BLE001 - re-raised on the calling thread
                result['error'] = e
            finally:
                done.set()

        log.info('%s: downloading %d files (%.1f GB) from %s@%s', model, len(todo), total / 1e9, s.hf_repo, s.hf_revision)
        worker = threading.Thread(target=run, name='decider-mlx-fetch', daemon=True)
        worker.start()
        progress = Progress(total, start=existing)
        if screen is not None:
            screen.download(model, progress)          # the first tip at once
        last_log = time.monotonic()
        try:
            while not done.wait(0.25):
                progress.update(counter.value)
                if screen is not None:
                    screen.download(model, progress)
                elif time.monotonic() - last_log > 10:
                    last_log = time.monotonic()
                    log.info('%s: %s', model, progress.text())
        except BaseException:
            cancel.set()
            worker.join(timeout=5)
            if screen is not None:
                screen.close()
            raise
        if 'error' in result:
            if screen is not None:
                screen.close()
            raise _hub_error(model, result['error']) from result['error']
        verified = result.get('verified', set())
        progress.update(total)
        if screen is not None:
            screen.download(model, progress, final=True)
        still = missing_files(root, manifest)
        if still:
            raise FetchError(f'{model}: {len(still)} files are missing after the download, for example {still[0]}')
    if verify:
        verify_files(model, root, manifest, screen, verified=verified)
    return root


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_files(model, root, manifest=None, screen=None, verified=None):
    """Check every manifest file's size and sha256. Raises FetchError naming the files that differ.
    Files named in `verified` were already hash-checked during this fetch and are not re-read."""
    manifest = manifest or read_manifest(Path(root) / MANIFEST)
    files = manifest['files']
    total = sum(e['size'] for e in files.values())
    bad = missing_files(root, manifest)
    already = set(verified) if verified else set()
    done = 0
    progress = Progress(total)
    for i, (name, entry) in enumerate(sorted(files.items())):
        if name not in bad and name not in already and sha256(Path(root) / name) != entry['sha256']:
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

    def __init__(self, total, start=0):
        """`start`: bytes already on disk before the download (not counted in the rate or the downloaded size)."""
        self.total, self.t0 = max(1, total), time.monotonic()
        self.start = self.done = min(self.total, max(0, start))
        self._samples = [(self.t0, self.done)]

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
    the block is redrawn in place; otherwise a progress line is written every 10 s and each tip once. The tips are in
    Japanese or English (tips.language())."""

    def __init__(self, stream=None, every=8.0, width=100, lang=None):
        from .tips import language
        self.stream = stream or sys.stderr
        self.lang = lang or language()
        self.tty = self.stream.isatty()
        self.every, self.width = every, width
        self.t0 = time.monotonic()
        self.lines = 0
        self.last_plain = 0.0
        self.tip_shown = -1

    def _tip_index(self):
        return int((time.monotonic() - self.t0) // self.every)

    def _block(self, head, model):
        from .tips import LABEL, count, tip
        i = self._tip_index()
        title, lines = tip(i, model, self.lang)
        n = count(self.lang)
        out = [head, '', f'  {LABEL[self.lang]} {i % n + 1}/{n}: {title}']
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
            got = (progress.total - progress.start) / 1e9
            head = f'Downloaded {model}: {got:.1f} GB in {_duration(time.monotonic() - progress.t0)}'
        if self.tty:
            block, _ = self._block(head, model)
            self._draw([head] if final else block)
            if final:
                self.lines = 0
            return
        now = time.monotonic()
        block, i = self._block(head, model)
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

    def next_steps(self, model):
        """The command to run next, shown once when `decider-mlx fetch` is done."""
        from .tips import next_steps
        self.stream.write('\n' + '\n'.join(next_steps(model, self.lang)) + '\n')
        self.stream.flush()

    def close(self):
        self.lines = 0

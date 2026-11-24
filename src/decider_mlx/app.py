"""The App: one worker thread owns MLX and runs every request in arrival order (FIFO).

`load(model)` returns when the weights are loaded. The startup preparation (synthetic warmup requests, tracing the
bucket graphs, the bulk graphs) then runs on the same worker between requests: before each preparation step the worker
waits up to GRACE_S for a request and serves every queued request first. Answers never depend on the preparation.
"""
from __future__ import annotations

import gc
import logging
import mmap
import os
import queue
import sys
import threading
import time

log = logging.getLogger(__name__)
GRACE_S = 0.02      # after a request, wait this long for the next one before a preparation step
BUSY_S = 1.0        # ... while the last request finished less than this ago (otherwise run the step at once)


def _filler(tokens):
    # Synthetic text, not from any dataset (about 1.3 words per token in this tokenizer family).
    words = ('alpha river stone window garden metal paper cloud orange signal market valley '
             'engine forest silver number bridge planet coffee summer').split()
    return ' '.join(words[i % len(words)] + ('.' if i % 13 == 12 else '') for i in range(int(tokens * 0.78)))


def _q(i, kind='noul'):
    if kind == 'choice':
        return {'type': 'choice', 'instructions': f'Which word fits item {i}?', 'criteria': {'a': 'alpha', 'b': 'river'}}
    if kind == 'score':
        return {'type': 'score', 'instructions': f'Is item {i} mentioned?', 'criteria': ['No', 'Yes']}
    return {'type': 'noul', 'instructions': f'Does the text mention item {i}?'}


def _image():
    from PIL import Image
    return Image.new('RGB', (448, 336), (120, 90, 60))


def warmup_requests(kind):
    """Synthetic startup requests: [(name, request(s), options)]. They run as ordinary requests."""
    if kind == 'flash':
        def req(tokens, kinds=('noul',)):
            return {'state': _filler(tokens), 'questions': {f'w{j}': _q(j, k) for j, k in enumerate(kinds)}}
        cases = [('single', lambda: req(300), {}),
                 ('true_batch', lambda: [req(150 + 9 * i) for i in range(20)], {}),
                 ('long', lambda: req(3500), {}),
                 ('multi_question', lambda: req(250, ('noul', 'noul', 'noul')), {}),
                 ('choice_score', lambda: req(280, ('noul', 'choice', 'score')), {}),
                 ('raw', lambda: req(260), {'raw': True}),
                 ('image', lambda: {'state': '', 'images': [_image()], 'questions': {'c': _q(0, 'choice'), 'n': _q(1), 's': _q(2, 'score')}}, {})]
        out = [(f'{name}-{mode}', make, dict(kw, mode=mode)) for mode in ('separate', 'isolated') for name, make, kw in cases]
        for kinds in [('choice', 'score'), ('choice', 'noul'), ('score',), ('score', 'score'), ('noul', 'score'), ('noul', 'noul')]:
            for mode in ('separate', 'isolated'):
                out.append((f'pair-{"-".join(kinds)}-{mode}', lambda kinds=kinds: {'state': _filler(200), 'questions': {f'w{j}': _q(j, k) for j, k in enumerate(kinds)}}, dict(mode=mode)))
        image_q = {'choice': {'type': 'choice', 'instructions': 'Which word fits?', 'criteria': {'a': 'alpha', 'b': 'river'}},
                   'score': {'type': 'score', 'instructions': 'Is it shown?', 'criteria': ['No', 'Yes']},
                   'noul': {'type': 'noul', 'instructions': 'Is a river shown?'}}
        for kinds in [('noul',), ('choice',), ('score',), ('choice', 'noul'), ('choice', 'score')]:
            for mode in ('separate', 'isolated'):
                out.append((f'image-{"-".join(kinds)}-{mode}', lambda kinds=kinds: {'state': 'Look at the image.', 'images': [_image()],
                            'questions': {f'w{j}': image_q[k] for j, k in enumerate(kinds)}}, dict(mode=mode)))
        return out
    text = ("The museum opened in 1931 and moved to its current building on the river in 1978. "
            "Its collection covers maps, clocks and navigation instruments from four centuries. ") * 3
    q = 'Did the museum move after 1950?'
    mixed = {'state': text, 'questions': {
        'noul': {'type': 'noul', 'instructions': q},
        'choice': {'type': 'choice', 'instructions': q, 'criteria': {'yes': 'Yes', 'no': 'No'}},
        'score': {'type': 'score', 'instructions': q, 'criteria': ['No', 'Yes']}}}
    single = {'state': text, 'questions': {'answer': {'type': 'noul', 'instructions': q}}}
    locate = {'state': text, 'questions': {'span': {'type': 'locate', 'instructions': 'When did the museum open?'},
                                           'has': {'type': 'noul', 'instructions': 'Is an opening year given?'}}}
    return [('single', lambda: single, {}), ('list2', lambda: [single, mixed], {}),
            ('list2_iso', lambda: [single, mixed], {'mode': 'isolated'}), ('raw', lambda: mixed, {'raw': True}),
            ('locate', lambda: locate, {})]


def bulk_warmup(kind):
    """One packed list per bulk domain, so those graphs are traced and run once."""
    reqs = [{'state': _filler(120 + 7 * i), 'questions': {'a': _q(i, ('noul', 'choice', 'score')[i % 3])}} for i in range(24)]
    out = [('bulk', reqs)]
    if kind == '27b':
        out.append(('bulk-locate', [{'state': _filler(200), 'questions': {'s': {'type': 'locate', 'instructions': 'Where is the river mentioned?'}}}]))
    return out


class _Job:
    __slots__ = ('fn', 'done', 'result', 'error')

    def __init__(self, fn):
        self.fn, self.done, self.result, self.error = fn, threading.Event(), None, None


class App:
    """A loaded model. decide() is thread-safe: calls are queued to the one worker thread in arrival order."""

    def __init__(self, model, _weights=None):
        from .registry import resolve_weights, spec
        self.spec = spec(model)
        self.model = self.spec.name
        self._weights = _weights or resolve_weights(self.model)   # _weights: internal (tests)
        self._jobs = queue.Queue()
        self._loaded = threading.Event()
        self._ready = threading.Event()
        self._load_error = None
        self.prep_error = None
        self.started = time.perf_counter()
        self.load_s = self.prep_s = None
        self.prep_total = 0
        self.prep_done = 0
        self.engine = None
        self._listeners = []
        self._last_served = 0.0
        self._worker = threading.Thread(target=self._run, name=f'decider-mlx-{self.model}', daemon=True)
        self._worker.start()
        self._loaded.wait()
        if self._load_error is not None:
            raise self._load_error

    # ---- public ----
    def decide(self, requests, *, execution_mode='separate', temperature=None, return_logits=False):
        """One request (dict) -> one response; a list -> a list of responses (raises on the first invalid request).
        temperature None uses the model's saved temperature. return_logits gives the raw readout logits per question."""
        from .engine import check_options
        check_options(execution_mode, temperature)
        return self._submit(lambda: self.engine.decide(requests, mode=execution_mode, temperature=temperature, raw=return_logits))

    def decide_each(self, requests, *, execution_mode='separate', temperature=None, return_logits=False):
        """A list of requests -> a list with a response or an exception (e.g. RequestError) per request."""
        from .engine import check_options
        check_options(execution_mode, temperature)
        return self._submit(lambda: self.engine.decide(list(requests), mode=execution_mode, temperature=temperature, raw=return_logits, strict=False))

    def decide_records(self, records):
        """records: [{'request', 'mode', 'temperature', 'raw'}] with per-record options -> a response or an exception each."""
        from .engine import check_options
        for r in records:
            check_options(r['mode'], r['temperature'])
        return self._submit(lambda: self.engine.decide_records(records))

    def wait_ready(self, timeout=None):
        """Block until the startup preparation finishes. Returns True only when preparation succeeded."""
        if not self._ready.wait(timeout):
            return False
        return self.prep_error is None

    @property
    def ready(self):
        return self._ready.is_set() and self.prep_error is None

    def status(self):
        return dict(model=self.model, loaded=self._loaded.is_set(), ready=self.ready, prep_done=self.prep_done,
                    prep_total=self.prep_total, load_s=self.load_s, prep_s=self.prep_s,
                    prep_error=None if self.prep_error is None else f'{type(self.prep_error).__name__}: {self.prep_error}')

    def metrics(self):
        def memory():
            import psutil
            m = self.engine.memory()
            m['rss_bytes'] = psutil.Process(os.getpid()).memory_info().rss
            return m
        # warm_done / startup_s / model_load_s: the names the R-000 benchmarks read
        return dict(self.status(), warm_done=self.ready, startup_s=self.load_s, model_load_s=self.load_s, **self._submit(memory))

    def on_progress(self, fn):
        """fn(app) is called on the worker thread after loading and after each preparation step."""
        self._listeners.append(fn)

    # ---- worker ----
    def _submit(self, fn):
        if threading.current_thread() is self._worker:
            return fn()
        job = _Job(fn)
        self._jobs.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

    def _serve(self, job):
        try:
            job.result = job.fn()
        except BaseException as e:   # noqa: BLE001 - handed back to the caller
            job.error = e
        finally:
            self._last_served = time.perf_counter()
            job.done.set()

    def _notify(self):
        for fn in self._listeners:
            try:
                fn(self)
            except Exception:   # noqa: BLE001 - a status display must never stop the worker
                log.exception('progress listener failed')

    def _run(self):
        try:
            if 'mlx.core' in sys.modules and os.environ.get('MLX_ENABLE_TF32') != '0':
                raise RuntimeError('Import decider_mlx before mlx (it sets MLX_ENABLE_TF32=0 for FP32 accumulation)')
            from .engine import Engine
            self.engine = Engine(self.spec, self._weights)
            self.load_s = time.perf_counter() - self.started
            log.info('%s: weights loaded in %.1f s', self.model, self.load_s)
        except BaseException as e:   # noqa: BLE001
            self._load_error = e
            self._loaded.set()
            return
        self._loaded.set()
        if self.spec.prefetch:
            threading.Thread(target=self._prefetch, daemon=True).start()
        step_name = 'init'
        try:
            steps = self._prep_steps()
            self._notify()
            while True:
                self._drain(GRACE_S if time.perf_counter() - self._last_served < BUSY_S else 0)
                try:
                    step_name, step = next(steps)
                except StopIteration:
                    break
                step()
                self.prep_done += 1
                self._notify()
        except FloatingPointError as e:
            self.prep_error = e
            log.error(
                '%s: non-finite readout during startup (%s): %s. '
                'Verify cached weights with `decider-mlx fetch --model %s` and check GPU compatibility.',
                self.model, step_name, e, self.model,
            )
        except BaseException as e:   # noqa: BLE001 - preparation is an optimization; requests still work
            self.prep_error = e
            log.exception('%s: startup preparation failed (%s)', self.model, step_name)
        if self.prep_error is None:
            self.prep_done = self.prep_total
            self.prep_s = time.perf_counter() - self.started
            log.info('%s: preparation done in %.1f s', self.model, self.prep_s)
        self._ready.set()
        self._notify()
        while True:
            self._serve(self._jobs.get())

    def _drain(self, grace):
        while True:
            try:
                job = self._jobs.get(timeout=grace) if grace > 0 else self._jobs.get_nowait()
            except queue.Empty:
                return
            self._serve(job)

    def _prefetch(self):
        """Read every page of the mapped main weights once (madvise WILLNEED + one touch per 16 KiB page)."""
        import numpy as np
        for owner, _, _ in list(self.engine.loader.files.values()):
            try:
                owner._mmap.madvise(mmap.MADV_WILLNEED)
            except (AttributeError, OSError, ValueError):
                pass
            np.bitwise_xor.reduce(owner[::2048])

    def _prep_steps(self):
        """The startup preparation as (name, callable) steps; prep_total is set before the first step."""
        import mlx.core as mx
        engine, kind = self.engine, self.spec.warmup
        steps = []
        for name, make, opts in warmup_requests(kind):
            def run(make=make, opts=opts):
                engine.decide(make(), mode=opts.get('mode', 'separate'), temperature=1.0, raw=opts.get('raw', False))
                mx.synchronize()
            steps.append(('warmup:' + name, run))
        if self.spec.image_graphs:
            steps.append(('image-graphs', None))
        steps.append(('bucket-graphs', None))
        for name, reqs in bulk_warmup(kind):
            steps.append((name, lambda reqs=reqs: engine.bulk_records([dict(request=r, mode='separate', temperature=1.0, raw=False) for r in reqs], strict=True)))
        steps.append(('image-processor', engine.encoder.pre.load_processor))

        def cleanup():
            gc.collect(); mx.clear_cache(); mx.synchronize()
        steps.append(('cleanup', cleanup))
        # graph keys are counted now; keys traced meanwhile by real requests are skipped (and still counted as done)
        planned_images = list(engine.buckets.precompile_image_steps()) if self.spec.image_graphs else []
        self.prep_total = len(steps) - (2 if self.spec.image_graphs else 1) + len(planned_images) + self._count_bucket_steps()

        def generate():
            for name, fn in steps:
                if name == 'image-graphs':
                    for _, key, trace in engine.buckets.precompile_image_steps():
                        yield 'image:' + str(key), trace
                elif name == 'bucket-graphs':
                    for _, key, trace in engine.buckets.precompile_steps():
                        yield 'bucket:' + str(key), trace
                else:
                    yield name, fn
        return generate()

    def _count_bucket_steps(self):
        b = self.engine.buckets
        n = 0
        for structure, return_all, locate, score in b.precompile:
            for bucket in b.buckets:
                total, share = b._share(bucket, structure)
                if total <= 0:
                    continue
                if max(share) > 512:
                    break
                n += 1
        return n


def load(model, **internal):
    """Load `model` ('pplx-decider-flash-v1.3' or 'pplx-decider-v1.3') and return an App once the weights are loaded.
    Fetches the weights from Hugging Face first when they are not in the local cache."""
    return App(model, **internal)

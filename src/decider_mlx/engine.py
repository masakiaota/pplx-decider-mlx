"""The inference engine: requests -> packed forwards -> answers. Runs on one thread (the App worker).

Routes (all fixed; the per-model differences come from registry.ModelSpec):
  - decide(list) with >= BULK_MIN requests: fixed 2048-token packing (batching/bulk.py)
  - otherwise each request is planned into sequences (one per question, or one per request in isolated mode); the
    sequences are sorted by length and grouped (at most `batch_size` sequences and 8192 padded tokens per forward)
  - a forward of short sequences runs a length-bucketed graph traced at startup (batching/bucket.py); other shapes run
    a per-shape compiled graph or the eager forward
The answer of a request never depends on the time it arrives or on the startup preparation.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from . import registry
from .batching.bucket import Buckets
from .batching.bulk import Bulk
from .model import METAL_ATTENTION_LIMIT, AffineInt8, Backbone, Layout, MappedSafetensors, W4A8, forward_core, gpu_info, has_m5_kernels
from .protocol import Encoder, RequestError, answer, probabilities, validate

log = logging.getLogger(__name__)

MAX_LENGTH = 32768        # the official DEFAULT_MAX_LENGTH: longer inputs raise LengthError, never truncated
BULK_MIN = 64             # decide(list) with at least this many requests uses the fixed-budget packing
BATCH_TOKENS = 8192
SHAPE_GRAPHS = 24         # compiled per-shape graphs kept per kind (oldest dropped; only speed depends on it)
CACHE_LIMIT = 1_000_000_000   # MLX buffer cache; does not change any value


class _LRU(dict):
    def __init__(self, size):
        super().__init__(); self.size = size

    def add(self, key, value):
        if len(self) >= self.size:
            del self[next(iter(self))]
        self[key] = value
        return value


class Engine:
    def __init__(self, spec, weights):
        self.spec = spec
        self.directory = Path(weights.main)
        self.weights_paths = weights
        mx.set_cache_limit(CACHE_LIMIT)
        self.config = json.loads((self.directory / 'decision_config.json').read_text())
        protocol = json.loads((self.directory / 'joint_config.json').read_text())
        if protocol.get('version') != 4 or protocol.get('template') != 'joint-chunked-readout-v1':
            raise ValueError('The model must use Decider v1.3 protocol v4')
        if self.config.get('attention_mode') != 'noncausal_full_attention':
            raise ValueError('Expected the noncausal Decider classifier')
        self.temperature = float(self.config['temperature'])
        if abs(self.temperature - spec.temperature) > 1e-6:
            raise ValueError(f'{spec.name}: saved temperature {self.temperature} differs from the registry')
        self.model_name = spec.name
        self.max_length = MAX_LENGTH
        self.use_m5_kernels = has_m5_kernels()
        if spec.arithmetic == 'w4a8-s4' and not self.use_m5_kernels:
            device_name, arch, _ = gpu_info()
            raise RuntimeError(
                f'{spec.name} requires Apple silicon M5 or later (detected {device_name}, {arch}); '
                f'its row-wise 4-bit weights require M5 Metal tensor ops. Use pplx-decider-flash-v1.3 on M1-M4 Macs.'
            )
        if not self.use_m5_kernels:
            device_name, arch, _ = gpu_info()
            log.info('%s: detected %s (%s); using standard MLX quantized matmul and SDPA fallback', spec.name, device_name, arch)
        self.loader = MappedSafetensors()
        import time
        t = time.perf_counter(); self.load_times = {}
        self.encoder = Encoder(self.directory, self.config['codes'], self.config.get('max_pixels', 1638400))
        self.load_times['tokenizer'] = time.perf_counter() - t; t = time.perf_counter()
        self.backbone = model = Backbone(self.directory, self.loader, delta_threadgroup=spec.delta_threadgroup, use_nax=self.use_m5_kernels)
        self.load_times['backbone'] = time.perf_counter() - t; t = time.perf_counter()
        if spec.arithmetic == 'affine-int8':
            int8_values = self._int8_values() if self.use_m5_kernels else {}
            model.linear = self.arith = AffineInt8(model, int8_values, spec.score_layers, use_int8=self.use_m5_kernels)
        else:
            model.linear = self.arith = W4A8(model, self._rowq4(Path(weights.rowq4)))
        self.load_times['arithmetic'] = time.perf_counter() - t
        self.vision = None
        self.locate = None
        self.buckets = Buckets(self, registry.kernel_data(spec.name))
        self.bulk = Bulk(self)
        self.shape_graphs = _LRU(SHAPE_GRAPHS * 3)
        self.image_graphs = _LRU(SHAPE_GRAPHS)
        self._batch16 = False      # this call: noul/choice/score text list (Flash)
        self._long = False         # this call: one text request (27B long graph)
        mx.synchronize()

    # ---- weights ----
    def _int8_values(self):
        """Flash INT8 operands: the affine8 codes as u8, the per-row scale and the per-row code sums."""
        w = self.backbone.weights
        values = {}
        for key, sums in mx.load(str(self.directory / 'row-sums.safetensors')).items():
            name = key[:-8]
            row = w[name + '.scales'][:, 0].astype(mx.float32)
            mx.eval(row, sums)
            values[name] = (w[name + '.weight'], row, sums)
        return values

    def _rowq4(self, directory):
        rows = {}
        for path in sorted(directory.glob('*.safetensors')):
            name = path.name[:-len('.safetensors')]
            if name.split('.', 3)[-1] in registry.ROWQ4_SUFFIXES:
                d = self.loader(path)
                rows[name] = (d['codes'], d['row_scales'])
        if len(rows) != registry.ROWQ4_COUNT:
            raise ValueError(f'expected {registry.ROWQ4_COUNT} row-Q4 tensors in {directory}, found {len(rows)}')
        if any(n + '.weight' in self.backbone.weights for n in rows):
            raise ValueError('the main weights contain group-Q4 copies of row-Q4 tensors')
        return rows

    def load_locate(self):
        if self.locate is None:
            from .locate import Locate
            self.locate = Locate(self, self.weights_paths.locate, self.weights_paths.locate_adapter)
        return self.locate

    def load_vision(self):
        if self.vision is None:
            from .vision import Vision
            self.vision = Vision(self.backbone, compiled=self.spec.image_graphs)
        return self.vision

    # ---- requests ----
    def decide(self, requests, *, mode='separate', temperature=None, raw=False, strict=True):
        """requests: one request (dict) or a list. A list returns a list; with strict=False an invalid request gives
        its exception in place of its response instead of raising."""
        if isinstance(requests, dict):
            self._long = not requests.get('images')
            try:
                out = self._list([requests], mode, temperature, raw, strict)[0]
            finally:
                self._long = False
            if isinstance(out, Exception):
                raise out
            return out
        requests = list(requests)
        if len(requests) >= BULK_MIN:
            items = [dict(request=r, mode=mode, temperature=temperature, raw=raw) for r in requests]
            return self.bulk_records(items, strict)
        return self._list(requests, mode, temperature, raw, strict)

    def decide_records(self, records):
        """records: [dict(request, mode, temperature, raw)], each with its own options. Returns a response or an
        Exception per record. Long lists are packed (bulk); shorter ones run one request at a time."""
        if len(records) >= BULK_MIN:
            return self.bulk_records(records, strict=False)
        out = []
        for r in records:
            try:
                out.append(self.decide(r['request'], mode=r['mode'], temperature=r['temperature'], raw=r['raw']))
            except Exception as e:
                out.append(e)
        return out

    def bulk_records(self, records, strict):
        out = [None] * len(records)

        def write(i, response=None, error=None):
            if error is not None and strict:
                raise error
            out[i] = error if error is not None else response
        self.bulk.run(records, write, lambda request, mode, temperature, raw: self.decide(request, mode=mode, temperature=temperature, raw=raw), strict)
        return out

    def _list(self, batch, mode, temperature, raw, strict):
        """The per-request route for a list of requests (< BULK_MIN). Returns one response or Exception each."""
        out = [None] * len(batch)

        def fail(i, e):
            if strict:
                raise e
            out[i] = e
        for i, request in enumerate(batch):
            try:
                validate(request)
                if raw and any(q['type'] == 'locate' for q in request['questions'].values()):
                    raise RequestError('Raw decision logits do not apply to Locate')
            except Exception as e:
                fail(i, e)
        good = [i for i in range(len(batch)) if out[i] is None]
        kinds = {q['type'] for i in good for q in batch[i]['questions'].values()}
        # Flash: a list of text noul/choice/score requests runs up to 16 sequences per compiled forward
        self._batch16 = (self.spec.batch16 and len(good) > 1 and bool(kinds) and kinds <= {'noul', 'choice', 'score'}
                         and not any(batch[i].get('images') for i in good))
        batch_size = 16 if self._batch16 else 8
        try:
            scale = self.temperature if temperature is None else float(temperature)
            if 'locate' in kinds:
                self.load_locate()
            planned = {}
            for i in good:
                request = batch[i]
                try:
                    decisions = {k: q for k, q in request['questions'].items() if q['type'] != 'locate'}
                    located = [(k, q) for k, q in request['questions'].items() if q['type'] == 'locate']
                    plans = self.encoder.plan({**request, 'questions': decisions}, mode, self.max_length) if decisions else []
                    lplans = self.locate.plans(request['state'], located) if located else []
                    planned[i] = (plans, lplans)
                except Exception as e:
                    fail(i, e)
            order = [i for i in good if i in planned]
            answers = {i: {} for i in order}
            usage = {i: dict(input_tokens=0, output_tokens=0, sequences=0) for i in order}
            decided = [i for i in order if planned[i][0]]
            if decided:
                self._decisions([(i, planned[i][0], batch[i]) for i in decided], scale, raw, batch_size, answers)
            for i in decided:
                usage[i]['input_tokens'] += sum(c.length for c in planned[i][0]); usage[i]['sequences'] += len(planned[i][0])
            pending = [(i, plan) for i in order for plan in planned[i][1]]
            if pending:
                self._locate(pending, batch, batch_size, answers, usage)
        finally:
            self._batch16 = False
        for i in order:
            if raw:
                out[i] = answers[i]
            else:
                out[i] = {'model': self.model_name, 'answers': {key: answers[i][key] for key in batch[i]['questions']}, 'usage': usage[i]}
        return out

    @staticmethod
    def _groups(items, length, batch_size):
        items = sorted(items, key=length, reverse=True)
        groups, group = [], []
        for item in items:
            maximum = max([length(g) for g in group] + [length(item)])
            if group and (len(group) == batch_size or maximum * (len(group) + 1) > BATCH_TOKENS):
                groups.append(group); group = []
            group.append(item)
        if group:
            groups.append(group)
        return groups

    def _decisions(self, items, scale, raw, batch_size, answers):
        features = {}
        pictured = [(i, plans[0]) for i, plans, _ in items if plans[0].pixels is not None]
        if pictured:
            vision = self.load_vision()
            # Pack vision projections while keeping image-wise attention, at most 8192 patches per call.
            vision_groups, current, count = [], [], 0
            for item in pictured:
                n = len(item[1].pixels)
                if current and count + n > 8192:
                    vision_groups.append(current); current, count = [], 0
                current.append(item); count += n
            if current:
                vision_groups.append(current)
            merge = self.backbone.config['vision_config']['spatial_merge_size']
            for group in vision_groups:
                values = vision(np.concatenate([c.pixels for _, c in group]), np.concatenate([c.grid for _, c in group]))
                offset = 0
                for i, chunk in group:
                    n = int(np.prod(chunk.grid, axis=-1).sum()) // (merge * merge)
                    features[i] = values[offset:offset + n]; offset += n
        requests = {i: request for i, _, request in items}
        pending = [(i, chunk) for i, plans, _ in items for chunk in plans]
        for group in self._groups(pending, lambda item: item[1].length, batch_size):
            chunks = [chunk for _, chunk in group]
            multimodal = None
            if any(chunk.pixels is not None for chunk in chunks):
                multimodal = [self.vision.inputs(chunk, features[i]) if chunk.pixels is not None else None for i, chunk in group]
            values = self.forward(chunks, multimodal)
            mx.eval(values)
            values = np.asarray(values)
            if not np.isfinite(values).all():
                raise FloatingPointError('Non-finite readout')
            for row, (i, chunk) in enumerate(group):
                for question, (key, count) in enumerate(zip(chunk.keys, chunk.counts)):
                    scores = values[row, question, :count]
                    answers[i][key] = scores.tolist() if raw else answer(requests[i]['questions'][key], probabilities(scores, scale))

    def _locate(self, pending, batch, batch_size, answers, usage):
        self.backbone.locate_enabled = True
        try:
            for group in self._groups(pending, lambda item: item[1][0].length, batch_size):
                chunks = [plan[0] for _, plan in group]
                hidden = self.forward(chunks, return_all=True)
                mx.eval(hidden)
                for row, (i, (encoded, positions, offsets)) in enumerate(group):
                    answers[i].update(self.locate.answers(hidden[row], encoded, positions, offsets, batch[i]['state']))
                    usage[i]['input_tokens'] += encoded.length; usage[i]['sequences'] += 1
        finally:
            self.backbone.locate_enabled = False

    # ---- forwards ----
    def forward(self, chunks, multimodal=None, return_all=False):
        """(chunks, questions, R) readout logits, or (chunks, longest, D) hidden states with return_all."""
        arith = self.arith
        if isinstance(arith, AffineInt8) and arith.use_int8:
            flags = tuple(bool(getattr(c, 'score', False)) for c in chunks)
            if any(flags):
                arith.score, arith.mixed, arith.lengths = flags, not all(flags), tuple(c.length for c in chunks)
            try:
                return self._route(chunks, multimodal, return_all)
            finally:
                arith.score, arith.mixed, arith.lengths = None, False, ()
        return self._route(chunks, multimodal, return_all)

    def _route(self, chunks, multimodal, return_all):
        model, spec = self.backbone, self.spec
        n = len(chunks); lengths = [c.length for c in chunks]
        locate = model.locate_enabled
        if (self._batch16 and multimodal is None and not return_all and not locate and 8 < n <= 16
                and max(lengths) <= METAL_ATTENTION_LIMIT and sum(lengths) <= BATCH_TOKENS):
            return self._shape_graph(chunks, False)
        if multimodal is not None:
            if spec.image_graphs and not return_all:
                out = self.buckets.forward_images(chunks, multimodal)
                if out is not None:
                    return out
                if (all(v is not None for v in multimodal) and not locate and n <= 8
                        and (max(lengths) <= METAL_ATTENTION_LIMIT or (n == 1 and lengths[0] <= self.max_length))):
                    return self._image_graph(chunks, multimodal)
            return self._eager(chunks, multimodal, return_all)
        return self._text(chunks, return_all)

    def _text(self, chunks, return_all):
        out = self.buckets.forward_text(chunks, return_all)
        if out is not None:
            return out
        model, spec = self.backbone, self.spec
        n = len(chunks); lengths = [c.length for c in chunks]
        locate = model.locate_enabled
        if spec.long_graph_min == 0:      # Flash: per-shape graphs
            if locate and return_all:
                eligible = n <= 8 and (sum(lengths) <= BATCH_TOKENS or max(lengths) <= METAL_ATTENTION_LIMIT or (n == 1 and lengths[0] <= self.max_length))
            else:
                eligible = not locate and not return_all and n <= 8 and (max(lengths) <= METAL_ATTENTION_LIMIT or (n == 1 and lengths[0] <= self.max_length))
        else:                             # 27B: one long text request
            eligible = self._long and n == 1 and spec.long_graph_min < lengths[0] <= self.max_length
        if eligible:
            return self._shape_graph(chunks, return_all)
        return self._eager(chunks, None, return_all)

    def _eager(self, chunks, multimodal, return_all):
        ids = mx.array(np.concatenate([c.ids for c in chunks]))
        return forward_core(self.backbone, Layout(chunks, multimodal), ids, multimodal, return_all)

    def _shape_graph(self, chunks, return_all):
        """One compiled graph per sequence geometry (token ids are inputs)."""
        model = self.backbone
        key = (self.arith.key(), bool(model.locate_enabled), bool(return_all), tuple(tuple(c.segments) for c in chunks))
        graph = self.shape_graphs.get(key)
        if graph is None:
            geometry = [SimpleNamespace(length=c.length, segments=tuple(c.segments), keys=range(len(c.keys))) for c in chunks]
            layout = Layout(geometry, None)

            def run(ids):
                return forward_core(model, layout, ids, None, return_all, compiled=True)
            graph = self.shape_graphs.add(key, mx.compile(run))
        return graph(mx.array(np.concatenate([c.ids for c in chunks])))

    def _image_graph(self, chunks, multimodal):
        model = self.backbone
        key = (self.arith.key(), tuple(tuple(c.segments) for c in chunks))
        graph = self.image_graphs.get(key)
        if graph is None:
            geometry = [SimpleNamespace(length=c.length, segments=tuple(c.segments), keys=range(len(c.keys))) for c in chunks]
            n = len(chunks)

            def run(ids, *mm):
                pairs = [(mm[i], mm[n + i]) for i in range(n)]
                return forward_core(model, Layout(geometry, pairs), ids, pairs, False, compiled=True)
            graph = self.image_graphs.add(key, mx.compile(run))
        ids = mx.array(np.concatenate([c.ids for c in chunks]))
        return graph(ids, *[v[0] for v in multimodal], *[v[1] for v in multimodal])

    # ---- status ----
    def memory(self):
        return dict(load_times=self.load_times, active_bytes=mx.get_active_memory(), peak_bytes=mx.get_peak_memory(), cache_bytes=mx.get_cache_memory(),
                    mapped_bytes=self.loader.mapped_bytes)


def check_options(mode, temperature):
    if mode not in ('separate', 'isolated'):
        raise ValueError('execution_mode must be separate or isolated')
    if temperature is not None and not (isinstance(temperature, (int, float)) and math.isfinite(temperature) and temperature > 0):
        raise ValueError('temperature must be finite and positive')

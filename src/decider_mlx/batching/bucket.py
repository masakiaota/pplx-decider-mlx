"""Length-bucketed compiled graphs for short packed forwards (every sequence at most 512 tokens).

A forward is padded to a bucket length by appending ONE dummy sequence (its own segment: own attention range, own
DeltaNet state, no question, no marker). The graph takes every layout tensor as an input, so one trace serves every
real length that maps to the same (bucket, question structure, arithmetic). The dummy produces no readout row and is
dropped from Locate's hidden states. A bucket never spans a row count where MLX picks another matmul kernel
(data/boundaries-m5.json `edges`), so padding never changes a value. The graphs of the most common structures are
traced at startup (trace only, no execution); others trace on first use.
"""
from __future__ import annotations

import bisect
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from ..model import LAYOUT_TENSORS, METAL_ATTENTION_LIMIT, Layout, layers, locate_marker, logits, per_chunk

LIMIT = METAL_ATTENTION_LIMIT
IMAGE_MIN_TOTAL = 96      # image graphs traced at startup start at this many tokens


def _text_core(model, view, ids, return_all):
    x = locate_marker(model, model.weights['language_model.embed_tokens.weight'][ids], ids)
    x, pruned = layers(model, x, view, return_all)
    if return_all:
        return x
    return per_chunk(logits(model, x, view.markers, pruned), view.chunks)


def _image_core(model, view, x):
    x, pruned = layers(model, x, view, False)
    return per_chunk(logits(model, x, view.markers, pruned), view.chunks)


def _view(structure, tensors):
    static = [SimpleNamespace(keys=range(n)) for _, n in structure] + [SimpleNamespace(keys=())]
    return SimpleNamespace(chunks=static, lengths=[1] * len(static), **dict(zip(LAYOUT_TENSORS, tensors)))


def synthetic(structure, share, ids):
    """Sequences of the given structure and lengths (for tracing): up to 16-token questions, the rest context."""
    chunks = []
    for (nseg, nkeys), length in zip(structure, share):
        q = [min(16, max(1, (length - 1) // nseg))] * (nseg - 1); ctx = length - sum(q)
        if ctx < 1:
            return None
        chunks.append(SimpleNamespace(length=length, segments=(ctx, *q), keys=tuple(range(nkeys)), ids=ids(length)))
    return chunks


class Buckets:
    def __init__(self, engine, data):
        self.engine, self.model = engine, engine.backbone
        self.edges = sorted(data['edges'])
        self.buckets = sorted(data['buckets'])
        self.precompile = [(tuple(map(tuple, s)), bool(ra), bool(lo), tuple(sc) if sc else False) for s, ra, lo, sc in data['precompile']]
        self.precompile_images = [(tuple(map(tuple, s)), tuple(sc) if sc else False) for s, sc in data['precompile_images']]
        self.max_chunks = engine.spec.bucket_max_chunks
        self.dummy_id = int(engine.encoder.separator_id)
        self.text_graphs = {}
        self.image_graphs = {}

    def target(self, total):
        """The smallest bucket >= total + 1 (at least one dummy token) in the same kernel class as total, or None."""
        i = bisect.bisect_left(self.buckets, total + 1)
        if i == len(self.buckets):
            return None
        b = self.buckets[i]
        if bisect.bisect_right(self.edges, total) != bisect.bisect_right(self.edges, b):
            return None
        return b

    def _pad(self, chunks):
        total = sum(c.length for c in chunks)
        b = self.target(total)
        if b is None or b - total > LIMIT:
            return None
        pad = b - total
        return b, [*chunks, SimpleNamespace(length=pad, segments=(pad,), keys=(), ids=np.full(pad, self.dummy_id, np.int32))]

    # ---- text ----
    def forward_text(self, chunks, return_all):
        """Bucketed forward, or None when the shape is not bucketed."""
        model, arith = self.model, self.engine.arith
        if not (1 <= len(chunks) <= self.max_chunks and max(c.length for c in chunks) <= LIMIT
                and bool(return_all) == bool(model.locate_enabled)):
            return None
        if not return_all and getattr(arith, 'mixed', False):
            return self._split(chunks)
        padded = self._pad(chunks)
        if padded is None:
            return None
        b, padded = padded
        layout = Layout(padded, None)
        structure = tuple((len(c.segments), len(c.keys)) for c in chunks)
        graph = self._text_graph((b, structure, bool(return_all), bool(model.locate_enabled), arith.key()))
        out = graph(mx.array(np.concatenate([c.ids for c in padded])), *[getattr(layout, n) for n in LAYOUT_TENSORS])
        if return_all:
            return mx.stack([mx.pad(out[a:b2], [(0, max(c.length for c in chunks) - (b2 - a)), (0, 0)])
                             for a, b2 in zip(layout.offsets[:len(chunks)], layout.offsets[1:len(chunks) + 1])])
        return out[:len(chunks)]

    def _text_graph(self, key):
        graph = self.text_graphs.get(key)
        if graph is None:
            model = self.model; structure, return_all = key[1], key[2]

            def run(ids, *tensors):
                return _text_core(model, _view(structure, tensors), ids, return_all)
            graph = self.text_graphs[key] = mx.compile(run)
        return graph

    def _split(self, chunks):
        """Score and other sequences in one forward: a bucket graph is shared by every split of the same total, so the
        per-row score arithmetic cannot be part of it. Run the two groups as two forwards and restore the order."""
        arith = self.engine.arith
        saved = (arith.score, arith.mixed, arith.lengths)
        flags = saved[0]
        parts = []
        try:
            for flag in (False, True):
                idx = [i for i, f in enumerate(flags) if f == flag]; sub = [chunks[i] for i in idx]
                arith.score = (True,) * len(sub) if flag else None
                arith.mixed = False; arith.lengths = tuple(c.length for c in sub)
                parts.append((idx, self.engine._text(sub, False)))
        finally:
            arith.score, arith.mixed, arith.lengths = saved
        width = max(len(c.keys) for c in chunks); rows = [None] * len(chunks)
        for idx, out in parts:
            for j, i in enumerate(idx):
                r = out[j]
                rows[i] = r if r.shape[0] == width else mx.pad(r, [(0, width - r.shape[0]), (0, 0)])
        return mx.stack(rows)

    # ---- images (Flash) ----
    def forward_images(self, chunks, multimodal):
        """Bucketed forward of sequences with image embeddings, or None. The input embedding (text ids gathered, image
        features in place) is built outside the graph and passed in with the 3-axis positions."""
        model, arith = self.model, self.engine.arith
        if not (not model.locate_enabled and 1 <= len(chunks) <= self.max_chunks and max(c.length for c in chunks) <= LIMIT):
            return None
        if getattr(arith, 'mixed', False):
            return None
        padded = self._pad(chunks)
        if padded is None:
            return None
        b, padded = padded
        mm = [*multimodal, None]
        layout = Layout(padded, mm)
        table = model.weights['language_model.embed_tokens.weight']
        x = mx.concatenate([table[mx.array(c.ids)] if v is None else v[0][0] for c, v in zip(padded, mm)])
        structure = tuple((len(c.segments), len(c.keys)) for c in chunks)
        graph = self._image_graph((b, structure, arith.key()))
        return graph(x, *[getattr(layout, n) for n in LAYOUT_TENSORS])[:len(chunks)]

    def _image_graph(self, key):
        graph = self.image_graphs.get(key)
        if graph is None:
            model = self.model; structure = key[1]

            def run(x, *tensors):
                return _image_core(model, _view(structure, tensors), x)
            graph = self.image_graphs[key] = mx.compile(run)
        return graph

    # ---- startup: trace only ----
    def _share(self, b, structure):
        total = b - 1
        while total > 0 and self.target(total) != b:
            total -= 1
        return total, [total // len(structure) + (i < total % len(structure)) for i in range(len(structure))]

    def precompile_steps(self):
        """Startup steps: each traces one bucket graph (no execution), yields its key; keys traced meanwhile by real
        requests are skipped. The arithmetic state is restored before every yield."""
        model, arith = self.model, self.engine.arith
        rng = np.random.default_rng(0)
        ids = lambda n: rng.integers(1000, 30000, n).astype(np.int32)
        for structure, return_all, locate, score in self.precompile:
            if locate and model.locate is None:
                continue
            for b in self.buckets:
                key = (b, structure, return_all, locate, score)
                if key in self.text_graphs:
                    continue
                total, share = self._share(b, structure)
                if total <= 0:
                    continue
                if max(share) > LIMIT:
                    break
                chunks = synthetic(structure, share, ids)
                if not chunks:
                    continue
                yield ('bucket', key, lambda chunks=chunks, return_all=return_all, locate=locate, score=score:
                       self._trace_text(chunks, return_all, locate, score))

    def _trace_text(self, chunks, return_all, locate, score):
        model, arith = self.model, self.engine.arith
        saved_locate = model.locate_enabled
        model.locate_enabled = locate
        if score:
            arith.score, arith.mixed, arith.lengths = score, False, tuple(c.length for c in chunks)
        try:
            self.forward_text(chunks, return_all)      # traces the graph; the lazy result is dropped
        finally:
            model.locate_enabled = saved_locate
            if score:
                arith.score, arith.mixed, arith.lengths = None, False, ()

    def precompile_image_steps(self):
        model, arith = self.model, self.engine.arith
        D = model.text['hidden_size']
        for structure, score in self.precompile_images:
            for b in self.buckets:
                if b > LIMIT * len(structure) + LIMIT:
                    break
                key = (b, structure, score)
                if key in self.image_graphs:
                    continue
                total, share = self._share(b, structure)
                if total < IMAGE_MIN_TOTAL:
                    continue
                if max(share) > LIMIT:
                    break
                chunks = synthetic(structure, share, lambda n: np.zeros(n, np.int32))
                if not chunks:
                    continue
                mm = [(mx.zeros((1, c.length, D), model.dtype), mx.array(np.tile(np.arange(c.length, dtype=np.int32), (3, 1)))) for c in chunks]
                yield ('image', key, lambda chunks=chunks, mm=mm, score=score: self._trace_images(chunks, mm, score))

    def _trace_images(self, chunks, mm, score):
        arith = self.engine.arith
        if score:
            arith.score, arith.mixed, arith.lengths = score, False, tuple(c.length for c in chunks)
        try:
            self.forward_images(chunks, mm)
        finally:
            if score:
                arith.score, arith.mixed, arith.lengths = None, False, ()

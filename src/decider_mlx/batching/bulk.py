"""Long lists: fixed-token-budget packing inside the same decide() API.

decide(list) switches here at engine.BULK_MIN requests. Sequences are classified by arithmetic domain (text; Flash
also score; locate), sorted by length and first-fit packed into packs of exactly BUDGET tokens. One compiled graph per
domain serves every pack: the packing geometry is a graph input and unused tokens are independent dummy segments.
Every row is computed independently of its neighbours (INT8/W4A8 GEMMs accumulate integers per row; attention,
DeltaNet, convolution and norms are per-token or per-sequence kernels; the standard QMM always sees the same row
count), so a request's answer does not depend on the other requests or their order.
Sequences longer than SHORT_LIMIT tokens run one request at a time through the per-request route.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from ..model import AffineInt8, layers, logits
from ..protocol import RequestError, answer, probabilities, validate

BUDGET = 2048        # tokens per pack
WINDOW = 2048        # requests planned and packed at a time (bounds the image features held in memory)
SHORT_LIMIT = 512    # the segment attention kernel (used for every pack) covers sequences up to 512 tokens
FILLER = 512


class Unit:
    """One packed sequence (one encoded sequence of one request)."""
    __slots__ = ('rec', 'chunk', 'length', 'domain', 'x', 'pos', 'locate')

    def __init__(self, rec, chunk, domain, x=None, pos=None, locate=None):
        self.rec, self.chunk, self.length, self.domain = rec, chunk, chunk.length, domain
        self.x, self.pos, self.locate = x, pos, locate


class Bulk:
    def __init__(self, engine, budget=BUDGET):
        self.engine, self.model = engine, engine.backbone
        self.N = int(budget)
        self.S = self.N // 48 + self.N // FILLER + 2
        self.Q = max(128, -(-(self.N // 24) // 128) * 128)
        self.graphs = {}
        self.score_domain = isinstance(engine.arith, AffineInt8)

    # ---------- graphs ----------
    def _graph(self, domain):
        if domain in self.graphs:
            return self.graphs[domain]
        model = self.model
        return_all = domain == 'locate'

        def core(x, positions, ranges, starts, owners, blocks, prefixes, contexts, qoff, qstarts, qends, markers):
            layout = SimpleNamespace(starts=starts, owners=owners, blocks=blocks, prefixes=prefixes, contexts=contexts,
                                     question_offsets=qoff, question_starts=qstarts, question_ends=qends,
                                     markers=markers, ranges=ranges, positions=positions, lengths=[1])
            x, pruned = layers(model, x, layout, return_all)
            return x if return_all else logits(model, x, markers, pruned)
        self.graphs[domain] = mx.compile(core)
        return self.graphs[domain]

    def run_graph(self, domain, inputs):
        """Call the domain's graph; the first call traces it with the domain's arithmetic."""
        model, arith = self.model, self.engine.arith
        saved = model.locate_enabled
        model.locate_enabled = domain == 'locate'
        if self.score_domain and domain != 'locate':
            arith.bulk = domain
        try:
            return self._graph(domain)(*inputs)
        finally:
            model.locate_enabled = saved
            if self.score_domain:
                arith.bulk = None

    # ---------- packing ----------
    def pack_ffd(self, units):
        """First fit in the given (length-descending) order: every open pack is tried, so short units fill the gaps."""
        packs = []   # [units, tokens, questions]
        for u in units:
            q = len(u.chunk.keys)
            for p in packs:
                n = p[1] + u.length
                if n <= self.N and p[2] + q <= self.Q and len(p[0]) + 1 + (self.N - n + FILLER - 1) // FILLER <= self.S:
                    p[0].append(u); p[1] = n; p[2] += q
                    break
            else:
                packs.append([[u], u.length, q])
        return [p[0] for p in packs]

    def build(self, units):
        N, S, Q = self.N, self.S, self.Q
        model = self.model
        starts = np.full(S, N, np.int32); contexts = np.full(S, N, np.int32); qoff = np.zeros(S + 1, np.int32)
        qstarts = np.zeros(Q, np.int32); qends = np.zeros(Q, np.int32); markers = np.full(Q, N - 1, np.int32)
        owners = np.zeros(N, np.int32); blocks = np.zeros(N, np.int32); prefixes = np.zeros(N, np.int32)
        ranges = np.zeros((N, 4), np.int32); positions = np.zeros((3, N), np.int32); ids = np.zeros(N, np.int32)
        off, qn, s = 0, 0, 0
        images = []
        for u in units:
            c = u.chunk; L = u.length; start = off; ctx = start + c.segments[0]
            starts[s] = start; contexts[s] = ctx; owners[start:start + L] = s
            blocks[start:ctx] = start; prefixes[start:ctx] = start; ranges[start:ctx] = (start, start + L, 0, 0)
            ids[start:start + L] = c.ids
            pos = [np.arange(c.segments[0])]
            o = ctx
            for size in c.segments[1:]:
                qstarts[qn] = o; qends[qn] = o + size; markers[qn] = o + size - 1
                ranges[o:o + size] = (start, ctx, o, o + size); blocks[o:o + size] = o; prefixes[o:o + size] = ctx
                pos.append(np.arange(size) + c.segments[0])
                o += size; qn += 1
            qoff[s + 1] = qn
            if u.pos is not None:
                positions[:, start:start + L] = u.pos
                images.append((start, L, u.x))
            else:
                positions[:, start:start + L] = np.concatenate(pos)[None]
            off += L; s += 1
        # dummy filler: independent context-only segments, self-only attention
        while off < N:
            end = min(N, off + FILLER)
            starts[s] = off; contexts[s] = end; owners[off:end] = s; blocks[off:end] = off; prefixes[off:end] = off
            t = np.arange(off, end, dtype=np.int32)
            ranges[off:end, 0] = t; ranges[off:end, 1] = t + 1
            qoff[s + 1] = qn; off = end; s += 1
        qoff[s + 1:] = qn
        assert s <= S and qn <= Q
        mids = mx.array(ids)
        x = model.weights['language_model.embed_tokens.weight'][mids]
        if images:
            parts, cursor = [], 0
            for start, L, emb in images:
                if start > cursor:
                    parts.append(x[cursor:start])
                parts.append(emb); cursor = start + L
            if cursor < N:
                parts.append(x[cursor:])
            x = mx.concatenate(parts)
        if units[0].domain == 'locate':
            x = mx.where((mids == model.locate.marker)[:, None], model.locate.marker_embedding.astype(x.dtype), x)
        return [x, mx.array(positions), mx.array(ranges), mx.array(starts), mx.array(owners), mx.array(blocks),
                mx.array(prefixes), mx.array(contexts), mx.array(qoff), mx.array(qstarts), mx.array(qends), mx.array(markers)]

    # ---------- requests ----------
    def run(self, items, write, single, strict=False):
        """items: [dict(request, mode, temperature, raw)]. write(i, response=None, error=None).
        single(request, mode, temperature, raw): the per-request route (long sequences).
        strict: raise the first planning error (in input order) before any GPU work."""
        for s in range(0, len(items), WINDOW):
            self._window(items, list(range(s, min(len(items), s + WINDOW))), write, single, strict)

    def _window(self, items, todo, write, single, strict):
        engine = self.engine
        state = {}
        units = {'text': [], 'score': [], 'locate': []}
        fallback = []
        if any(q.get('type') == 'locate' for i in todo if isinstance(items[i]['request'], dict)
               for q in (items[i]['request'].get('questions') or {}).values() if isinstance(q, dict)):
            engine.load_locate()
        if any(isinstance(items[i]['request'], dict) and items[i]['request'].get('images') for i in todo):
            engine.load_vision()

        def plan_one(i):
            it = items[i]; req = it['request']
            validate(req)
            qs = req['questions']
            decisions = {k: q for k, q in qs.items() if q['type'] != 'locate'}
            located = [(k, q) for k, q in qs.items() if q['type'] == 'locate']
            if it['raw'] and located:
                raise RequestError('Raw decision logits do not apply to Locate')
            plans = engine.encoder.plan({**req, 'questions': decisions}, it['mode'], engine.max_length) if decisions else []
            lplans = engine.locate.plans(req['state'], located) if located else []
            return plans, lplans

        planned = {}
        for i in todo:
            try:
                planned[i] = plan_one(i)
            except Exception as e:
                if strict:
                    raise
                planned[i] = e

        for i in todo:
            p = planned.pop(i)
            try:
                if isinstance(p, Exception):
                    raise p
                plans, lplans = p
                if any(c.length > SHORT_LIMIT for c in plans) or any(q[0].length > SHORT_LIMIT for q in lplans):
                    fallback.append(i)
                    continue
                mine = []
                feats = None
                if plans and plans[0].pixels is not None:
                    feats = engine.vision(plans[0].pixels, plans[0].grid)
                for c in plans:
                    domain = 'score' if (self.score_domain and c.score) else 'text'   # per sequence, not per request
                    if feats is not None:
                        emb, pos = engine.vision.inputs(c, feats)
                        mine.append(Unit(i, c, domain, x=emb[0], pos=np.asarray(pos)))
                    else:
                        mine.append(Unit(i, c, domain))
                for q in lplans:
                    mine.append(Unit(i, q[0], 'locate', locate=q))
                temp = items[i]['temperature']
                scale = engine.temperature if temp is None else float(temp)
                state[i] = dict(request=items[i]['request'], raw=items[i]['raw'], scale=scale, answers={}, remaining=len(mine),
                                usage=dict(input_tokens=sum(u.length for u in mine), output_tokens=0, sequences=len(mine)))
                for u in mine:
                    units[u.domain].append(u)
            except Exception as e:
                if strict:
                    raise
                state.pop(i, None)
                write(i, error=e)

        def complete(i):
            st = state[i]
            st['remaining'] -= 1
            if st['remaining'] == 0:
                ans = {k: st['answers'][k] for k in st['request']['questions']}
                write(i, ans if st['raw'] else {'model': engine.model_name, 'answers': ans, 'usage': st['usage']})
                del state[i]

        def finish(item):
            domain, pack, out = item
            if domain == 'locate':
                start = 0
                for u in pack:
                    encoded, positions, offsets = u.locate
                    rec = state[u.rec]
                    rec['answers'].update(engine.locate.answers(out[start:start + u.length], encoded, positions, offsets, rec['request']['state']))
                    start += u.length
            else:
                values = np.asarray(out); q = 0
                for u in pack:
                    rec = state[u.rec]
                    for key, count in zip(u.chunk.keys, u.chunk.counts):
                        scores = values[q, :count]; q += 1
                        if not np.isfinite(scores).all():
                            raise FloatingPointError('Non-finite packed readout')
                        rec['answers'][key] = scores.tolist() if rec['raw'] else answer(rec['request']['questions'][key], probabilities(scores, rec['scale']))
            for u in pack:
                complete(u.rec)

        pending = None
        for domain in ('text', 'score', 'locate'):
            us = units[domain]
            us.sort(key=lambda u: -u.length)
            for pack in self.pack_ffd(us):
                out = self.run_graph(domain, self.build(pack))
                mx.async_eval(out)          # build the next pack while this one runs
                if pending is not None:
                    finish(pending)
                pending = (domain, pack, out)
        if pending is not None:
            finish(pending)
        for i in fallback:
            it = items[i]
            try:
                write(i, single(it['request'], it['mode'], it['temperature'], it['raw']))
            except Exception as e:
                if strict:
                    raise
                write(i, error=e)

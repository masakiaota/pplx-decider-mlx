"""Locate questions: query-conditioned verbatim span scoring on the backbone's hidden states.

Locate forwards use their own weights: Flash runs a merged 8-bit copy of the Locate-tuned linears (W + B@A*alpha/r,
quantized; `mlx_config.json` format decider-locate-merged-mlx-v1); 27B adds the rank-64 LoRA in BF16 at run time.
Both put the trained embedding on the <|locate|> marker token. Loaded on the first Locate question.
"""
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from .preprocess import Preprocessor
from .protocol import SYSTEM, MARKER, SEPARATOR, Encoded, LengthError, describe

TOKEN = '<|locate|>'
MARKER_EMBEDDING = 'language_model.embed_tokens.locate'


class LocateWeights:
    """What the backbone's linear needs for Locate forwards: merged weights (Flash) or the LoRA (27B)."""

    def __init__(self, model, folder, loader, adapter=None):
        folder = Path(folder)
        self.config = json.loads((folder / 'locate_config.json').read_text())
        if self.config['format'] != 'pplx-decider-locate-span-v1':
            raise ValueError('Unexpected Locate format')
        self.head = mx.load(str(folder / 'head.safetensors'))
        self.modules, self.weights, self.quantization, self.lora, self.lora_scale = frozenset(), None, None, {}, 1.0
        merged = folder / 'mlx_config.json'
        if merged.exists():
            marker = mx.load(str(folder / 'marker.safetensors'))
            if set(marker) != {MARKER_EMBEDDING}:
                raise ValueError('The merged Locate folder must hold only the marker embedding, no LoRA')
            self._load_merged(model, folder, loader)
            self.marker_embedding = marker[MARKER_EMBEDDING]
        else:
            self.lora = loader(adapter or folder / 'adapter.safetensors')
            self.lora_scale = self.config['lora']['alpha'] / self.config['lora']['rank']
            self.marker_embedding = self.lora[MARKER_EMBEDDING]
        self.marker = self.config['marker_id']
        mx.eval(self.lora, self.head, self.marker_embedding)

    def _load_merged(self, model, folder, loader):
        metadata = json.loads((folder / 'mlx_config.json').read_text())
        if metadata.get('format') != 'decider-locate-merged-mlx-v1' or metadata.get('lora_applied') is not True:
            raise ValueError('The Locate weights must declare the merged Locate LoRA')
        base = model.directory / 'mlx_config.json'
        revision = json.loads(base.read_text()).get('source_revision') if base.exists() else None
        if not revision or metadata.get('base_revision') != revision:
            raise ValueError('The merged Locate weights belong to a different base revision')
        modules = frozenset(metadata['merged_modules'])
        expected = frozenset(key.removesuffix('.weight') for key, value in model.weights.items()
                             if key.startswith('language_model.layers.') and key.endswith('.weight') and value.ndim == 2)
        if modules != expected:
            raise ValueError('The merged Locate module set differs from the backbone')
        merged = dict(model.weights)
        for name in modules:
            for suffix in ('weight', 'scales', 'biases'):
                merged.pop(name + '.' + suffix, None)
        allowed = {name + '.' + suffix for name in modules for suffix in ('weight', 'scales', 'biases')}
        loaded = set()
        index = json.loads((folder / 'model.safetensors.index.json').read_text())
        for filename in sorted(set(index['weight_map'].values())):
            values = loader(folder / filename)
            if not set(values) <= allowed:
                raise ValueError('Unexpected tensor in the merged Locate weights')
            for key, value in values.items():
                merged[key] = value.astype(model.dtype) if value.dtype in (mx.float16, mx.bfloat16, mx.float32) else value
                loaded.add(key)
            mx.eval([merged[key] for key in values])
        if any(name + '.weight' not in loaded for name in modules):
            raise ValueError('The merged Locate weights miss a linear weight')
        if frozenset(metadata['quantized_modules']) != frozenset(name for name in modules if name + '.scales' in loaded):
            raise ValueError('The merged Locate quantization metadata differs from the tensors')
        self.weights, self.modules, self.quantization = merged, modules, metadata['quantization']


class Locate:
    def __init__(self, engine, folder, adapter=None):
        model = engine.backbone
        self.engine = engine
        self.pre = Preprocessor(engine.directory, SYSTEM)   # own tokenizer: it gains the <|locate|> special token
        self.pre.add_special_token(TOKEN)
        weights = LocateWeights(model, folder, engine.loader, adapter)
        self.config, self.head = weights.config, weights.head
        if self.pre.token_id(TOKEN) != weights.marker:
            raise ValueError('The Locate marker differs from training')
        self.separator_id = self.pre.token_id(SEPARATOR)
        self.escapes = sorted({*self.pre.special_tokens(), MARKER, SEPARATOR, TOKEN}, key=len, reverse=True)
        model.locate = weights

    def text(self, value):
        text = describe(value)
        for token in self.escapes:
            text = text.replace(token, token.replace('<', '＜').replace('>', '＞'))
        return text

    def plans(self, text, items):
        """One packed sequence for the Locate questions of one request (split in halves while over max_length).
        Returns [(encoded, state token positions, their character offsets)]."""
        context = self.pre.chat_context('State:\n' + self.text(text))
        begin = len(context) - len(self.text(text))
        ids, mapping = self.pre.encode_offsets(context)
        prefix = ids + [self.separator_id]
        chunks = []
        for key, question in items:
            instructions = question.get('instructions')
            instructions = instructions if instructions is not None and instructions.strip() else key
            prompt = 'Question:\nLocate verbatim spans:\n' + self.text(instructions)
            chunks.append(self.pre.encode(prompt) + [self.engine.backbone.locate.marker])
        length = len(prefix) + sum(map(len, chunks))
        maximum = self.engine.max_length
        if length > maximum:
            if len(items) == 1:
                raise LengthError(f'Locate input needs {length} tokens, exceeding {maximum}; no truncation applied')
            midpoint = len(items) // 2
            return self.plans(text, items[:midpoint]) + self.plans(text, items[midpoint:])
        ids, markers = list(prefix), []
        for chunk in chunks:
            ids.extend(chunk)
            markers.append(len(ids) - 1)
        positions = [i for i, (a, b) in enumerate(mapping) if a < len(context) and b > begin and a < b]
        offsets = [(max(0, mapping[i][0] - begin), min(len(text), mapping[i][1] - begin)) for i in positions]
        encoded = Encoded(np.asarray(ids, np.int32), tuple(key for key, _ in items), (0,) * len(items),
                          (len(prefix), *map(len, chunks)), tuple(markers))
        return [(encoded, positions, offsets)]

    def _linear(self, x, name):
        result = x.astype(mx.float32) @ self.head[name + '.weight'].astype(mx.float32).T
        if name + '.bias' in self.head:
            result = result + self.head[name + '.bias'].astype(mx.float32)
        return result

    def spans(self, document, queries):
        ts, te = self._linear(document, 'ts'), self._linear(document, 'te')
        qs, qe, qp = (self._linear(queries, name) for name in ('qs', 'qe', 'qp'))
        scale = math.sqrt(self.config['head']['width'])
        starts = qs @ ts.T / scale + self.head['start_bias']
        ends = qe @ te.T / scale + self.head['end_bias']
        thresholds = self._linear(queries, 'thresholds')
        count = min(self.config['head']['proposals'], document.shape[0])
        result = []
        for i in range(queries.shape[0]):
            si, ei = mx.argsort(starts[i])[-count:], mx.argsort(ends[i])[-count:]
            width = ei[None, :] - si[:, None] + 1
            bucket = mx.clip(mx.floor(mx.log2(mx.maximum(width, 1).astype(mx.float32))), 0, 31).astype(mx.int32)
            scores = (ts[si] * mx.tanh(qp[i])) @ te[ei].T / scale
            scores = scores + starts[i, si][:, None] + ends[i, ei][None, :] + self.head['width_bias.weight'][bucket, 0]
            scores = mx.where((width > 0) & (width <= self.config['head']['max_span']), scores, -mx.inf)
            probabilities = mx.sigmoid(scores - thresholds[i, 2])
            mx.eval(probabilities, si, ei)
            probability = np.asarray(probabilities)
            rows, cols = np.nonzero(probability >= self.config.get('thresholds', {}).get('default', 0.5))
            a, b = np.asarray(si), np.asarray(ei)
            result.append([(int(a[row]), int(b[col]), float(probability[row, col])) for row, col in zip(rows, cols)])
        return result

    def answers(self, hidden, encoded, positions, offsets, text):
        """Span answers {key: {"type": "locate", "spans": [...]}} from one sequence's hidden states (T, D)."""
        raw = self.spans(hidden[mx.array(positions)], hidden[mx.array(encoded.markers)]) if positions else [[] for _ in encoded.keys]
        out = {}
        for key, spans in zip(encoded.keys, raw):
            unique = {}
            for first, last, score in spans:
                a, b = offsets[first][0], offsets[last][1]
                while a < b and text[a].isspace():
                    a += 1
                while a < b and text[b - 1].isspace():
                    b -= 1
                if a < b:
                    unique[a, b] = max(unique.get((a, b), 0.0), score)
            out[key] = {'type': 'locate', 'spans': [{'start': a, 'end': b, 'text': text[a:b], 'score': p} for (a, b), p in sorted(unique.items())]}
        return out

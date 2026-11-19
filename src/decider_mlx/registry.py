"""The two supported models and everything that differs between them.

A model is selected by its official name only. Everything not listed here is shared code.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).parent / 'data'


@dataclass(frozen=True)
class ModelSpec:
    name: str                       # official model name: CLI --model, HTTP "model", response "model"
    hf_repo: str                    # converted MLX weights, tokenizer, processor and configs
    hf_revision: str                # pinned revision (commit) of hf_repo
    temperature: float              # saved temperature (decision_config.json), used when the request gives none
    arithmetic: str                 # 'affine-int8' (Flash) or 'w4a8-s4' (27B), see model.py
    score_layers: tuple = ()        # layers kept on the standard QMM for score sequences (affine-int8)
    delta_threadgroup: tuple = (32, 1, 1)   # DeltaNet kernel threadgroup (simdgroups per group never change values)
    bucket_max_chunks: int = 8      # most sequences in one bucketed short forward
    batch16: bool = False           # text lists of noul/choice/score: up to 16 sequences per compiled forward
    long_graph_min: int = 0         # compiled graph for one sequence longer than this (0: any single sequence;
                                    # otherwise only for a single text request)
    image_graphs: bool = False      # image forwards on bucket / per-shape graphs, vision blocks compiled
    join_image_text_state: bool = False   # HTTP: an array state of image parts + text parts becomes the joined text
    prefetch: bool = False          # read the mapped main weights once in a background thread after loading
    warmup: str = ''                # synthetic startup requests (app.py)
    layout: dict = field(default_factory=dict)  # files in the HF snapshot: main dir, Locate dir, extra weights
    recommended_memory_gb: int = 0


MODELS = {
    'pplx-decider-flash-v1.3': ModelSpec(
        name='pplx-decider-flash-v1.3',
        hf_repo='masakiaota/pplx-decider-flash-v1.3-mlx',
        hf_revision='TBD-after-upload',     # TODO(stage 4): fill in the commit hash after the upload
        temperature=1.3302962,
        arithmetic='affine-int8', score_layers=tuple(range(16)), delta_threadgroup=(32, 4, 1),
        bucket_max_chunks=7, batch16=True, long_graph_min=0, image_graphs=True, join_image_text_state=False,
        prefetch=True, warmup='flash',
        layout=dict(main='.', locate='locate'),
        recommended_memory_gb=24),
    'pplx-decider-v1.3': ModelSpec(
        name='pplx-decider-v1.3',
        hf_repo='masakiaota/pplx-decider-v1.3-mlx',
        hf_revision='TBD-after-upload',     # TODO(stage 4): fill in the commit hash after the upload
        temperature=1.2263882,
        arithmetic='w4a8-s4', delta_threadgroup=(32, 1, 1),
        bucket_max_chunks=8, batch16=False, long_graph_min=4096, image_graphs=False, join_image_text_state=True,
        prefetch=False, warmup='27b',
        layout=dict(main='.', locate='locate', rowq4='rowq4', locate_adapter='locate/adapter.safetensors'),
        recommended_memory_gb=36),
}

ROWQ4_SUFFIXES = ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'linear_attn.in_proj_qkv', 'linear_attn.in_proj_z',
                  'linear_attn.out_proj', 'self_attn.q_proj', 'self_attn.o_proj')
ROWQ4_COUNT = 368


class ModelNameError(ValueError):
    pass


def spec(name):
    if name not in MODELS:
        raise ModelNameError(f"Unknown model {name!r}; available: {', '.join(repr(m) for m in MODELS)}")
    return MODELS[name]


def kernel_data(name):
    """Length buckets, kernel edges and startup graph keys measured on M5 (data/boundaries-m5.json)."""
    return json.loads((DATA / 'boundaries-m5.json').read_text())[name]


@dataclass(frozen=True)
class Weights:
    main: Path                      # configs, tokenizer, processor, main weights, readout
    locate: Path                    # Locate head and config (and the merged Locate weights on Flash)
    rowq4: Path | None = None       # 27B: one safetensors per row-Q4 tensor (codes, row_scales)
    locate_adapter: Path | None = None   # 27B: the BF16 Locate LoRA


def weights_from_snapshot(model, root):
    root = Path(root); lay = MODELS[model].layout
    return Weights(main=root / lay['main'], locate=root / lay['locate'],
                   rowq4=root / lay['rowq4'] if 'rowq4' in lay else None,
                   locate_adapter=root / lay['locate_adapter'] if 'locate_adapter' in lay else None)


# --- internal, development only (removed with stage 4): read the weights from the research workspace -------------
_DEV_ENV = '_DECIDER_MLX_DEV_WEIGHTS'


def _dev_weights(model):
    root = os.environ.get(_DEV_ENV)
    if not root:
        return None
    root = Path(root)
    if model == 'pplx-decider-flash-v1.3':
        return Weights(main=root / 'experiments/flash-shared-q8/001-shared-affine8/weights',
                       locate=root / 'experiments/day-20261119/flash-default/weights/locate-merged')
    pack = root / 'experiments/night-20261119/B-002-no-groupq4/pack'
    return Weights(main=pack, locate=pack / 'locate-span', rowq4=root / 'experiments/night-20261119/E-004-w4a8-s4/rowq4-s4-d002',
                   locate_adapter=root / 'experiments/day-20261119/27b-default/weights/locate-adapter-bf16.safetensors')


def resolve_weights(model, fetch=True):
    """Local weight paths for `model`: the pinned Hugging Face snapshot (fetched when missing and fetch is true)."""
    dev = _dev_weights(model)
    if dev is not None:
        return dev
    from . import fetch as fetching
    return weights_from_snapshot(model, fetching.snapshot(model, download=fetch))

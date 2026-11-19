"""Download the converted weights from Hugging Face at the pinned revision (`decider-mlx fetch`).

serve / run / load() call `snapshot(model)`, which fetches the weights when they are not in the Hugging Face cache.
"""
from __future__ import annotations

import logging

from .registry import spec

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    pass


def snapshot(model, download=True):
    """Path of the local snapshot of `model`'s pinned revision; downloads it first when missing and download is true."""
    s = spec(model)
    if s.hf_revision.startswith('TBD'):
        # TODO(stage 4): remove once the repositories are uploaded and the revisions are pinned in registry.py
        raise FetchError(f'{model}: the Hugging Face revision is not pinned yet ({s.hf_repo})')
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(s.hf_repo, revision=s.hf_revision, local_files_only=True)
    except Exception:
        if not download:
            raise FetchError(f'{model} is not downloaded; run: decider-mlx fetch --model {model}') from None
    return fetch(model)


def fetch(model, progress=None):
    """Download `model` (TODO(stage 4): progress, remaining time and tips on the terminal)."""
    s = spec(model)
    if s.hf_revision.startswith('TBD'):
        raise FetchError(f'{model}: the Hugging Face revision is not pinned yet ({s.hf_repo})')
    from huggingface_hub import snapshot_download
    log.info('downloading %s from %s@%s', model, s.hf_repo, s.hf_revision)
    return snapshot_download(s.hf_repo, revision=s.hf_revision)

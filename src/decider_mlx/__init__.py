"""decider-mlx: MLX inference engine for Perplexity Decider v1.3 models on Apple silicon Macs.

    from decider_mlx import load
    app = load("pplx-decider-flash-v1.3")
    app.decide({"state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}})
"""
import os as _os
import sys as _sys

# M5 defaults to reduced-precision FP32 matrix products; the engine needs FP32 accumulation. MLX reads this when it
# is first imported, so decider_mlx must be imported before mlx.
if 'mlx.core' not in _sys.modules:
    _os.environ['MLX_ENABLE_TF32'] = '0'

__version__ = "0.1.0.dev0"

from .app import App, load  # noqa: E402
from .protocol import LengthError, RequestError  # noqa: E402
from .registry import MODELS, ModelNameError  # noqa: E402

__all__ = ['App', 'load', 'LengthError', 'RequestError', 'ModelNameError', 'MODELS']

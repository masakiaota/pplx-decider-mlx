import os

# M5 defaults to reduced-precision FP32 matrix products. The unoptimized baseline
# preserves FP32 accumulation; opt in explicitly when measuring that candidate.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

from .batched import Engine
from .protocol import LengthError, RequestError

__all__ = ["Engine", "LengthError", "RequestError"]

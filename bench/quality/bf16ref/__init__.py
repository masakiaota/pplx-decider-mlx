"""BF16 reference implementation of Perplexity Decider v1.3 in plain MLX, for quality measurements only.

It runs the original perplexity-ai checkpoints unquantized, so its outputs stand in for the original model's. It
was validated against the official PyTorch implementation (27B: 100/100 gate decisions, mean |dp| 0.0014; Flash:
99/100, the one difference at an official probability of exactly 0.5). It is slow and memory-hungry by design:
27B needs the whole BF16 model resident (MLX peak about 57 GB).
"""
import decider_mlx  # noqa: F401  (sets MLX_ENABLE_TF32=0 before mlx is imported: FP32 accumulation)

from .engine import Engine  # noqa: E402

__all__ = ['Engine']

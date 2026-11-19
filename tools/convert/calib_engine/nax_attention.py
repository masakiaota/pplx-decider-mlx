from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).parent
_kernel = mx.fast.metal_kernel(
    name="decider_nax_varlen", input_names=["queries", "keys", "values", "q_offsets", "k_offsets", "tile_samples", "tile_local"],
    output_names=["output"], source=(ROOT / "steel_body.metal").read_text(), header=(ROOT / "steel_header.metal").read_text(),
)


def attention(q, k, v, q_lengths, k_lengths):
    q_offsets = mx.array(np.cumsum([0, *q_lengths]), mx.int32)
    k_offsets = mx.array(np.cumsum([0, *k_lengths]), mx.int32)
    counts = [(length + 31) // 32 for length in q_lengths]
    samples = mx.array(np.repeat(np.arange(len(counts), dtype=np.int32), counts))
    local = mx.array(np.concatenate([np.arange(count, dtype=np.int32) for count in counts]))
    return _kernel(inputs=[q, k, v, q_offsets, k_offsets, samples, local],
                   template=[("Data", q.dtype), ("HeadDim", q.shape[2]), ("Heads", q.shape[1]), ("KVHeads", k.shape[1])],
                   grid=(sum(counts) * 128, q.shape[1], 1), threadgroup=(128, 1, 1),
                   output_shapes=[q.shape], output_dtypes=[q.dtype])[0]

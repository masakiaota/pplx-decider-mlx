"""Attention kernels for packed sequences.

segment_attention: one query row attends to up to two key ranges (its context and its own question), used for every
packed forward whose chunks are at most 512 tokens. tiled_attention: the MLX steel NAX attention kernel adapted to
variable-length query/key offsets (steel_header.metal / steel_body.metal, MIT, Apple Inc.; see NOTICE), used for
longer chunks.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np

_attention = mx.fast.metal_kernel(
    name="decider_segment_attention", input_names=["q", "k", "v", "ranges"], output_names=["out"],
    source=r"""
    const uint lane = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint query = row / H;
    const uint head = row % H;
    const uint kvhead = head / (H / HKV);
    float queries[D / 32];
    float values[D / 32];
    for (uint i = 0; i < D / 32; ++i) {
        queries[i] = float(q[(query * H + head) * D + lane + 32 * i]);
        values[i] = 0.0f;
    }
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint interval = 0; interval < 2; ++interval) {
    for (uint token = ranges[query * 4 + interval * 2]; token < uint(ranges[query * 4 + interval * 2 + 1]); ++token) {
        float score = 0.0f;
        for (uint i = 0; i < D / 32; ++i)
            score += queries[i] * float(k[(token * HKV + kvhead) * D + lane + 32 * i]);
        score = metal::simd_sum(score) * metal::rsqrt(float(D));
        float next_max = metal::max(maximum, score);
        float old_scale = metal::exp(maximum - next_max);
        float new_scale = metal::exp(score - next_max);
        denominator = denominator * old_scale + new_scale;
        for (uint i = 0; i < D / 32; ++i)
            values[i] = values[i] * old_scale + new_scale * float(v[(token * HKV + kvhead) * D + lane + 32 * i]);
        maximum = next_max;
    }
    }
    for (uint i = 0; i < D / 32; ++i)
        out[(query * H + head) * D + lane + 32 * i] = OUT(values[i] / denominator);
    """,
)


def attention(q, k, v, ranges):
    count, heads, width = q.shape
    return _attention(inputs=[q, k, v, ranges],
                      template=[("H", heads), ("HKV", k.shape[1]), ("D", width), ("OUT", q.dtype)],
                      grid=(32, count * heads, 1), threadgroup=(32, 1, 1),
                      output_shapes=[q.shape], output_dtypes=[q.dtype])[0]



def segment_attention(q, k, v, ranges):
    count, heads, width = q.shape
    return _attention(inputs=[q, k, v, ranges],
                      template=[("H", heads), ("HKV", k.shape[1]), ("D", width), ("OUT", q.dtype)],
                      grid=(32, count * heads, 1), threadgroup=(32, 1, 1),
                      output_shapes=[q.shape], output_dtypes=[q.dtype])[0]


_HERE = Path(__file__).parent
_tiled = mx.fast.metal_kernel(
    name="decider_nax_varlen", input_names=["queries", "keys", "values", "q_offsets", "k_offsets", "tile_samples", "tile_local"],
    output_names=["output"], source=(_HERE / "steel_body.metal").read_text(), header=(_HERE / "steel_header.metal").read_text(),
)


def tiled_attention(q, k, v, q_lengths, k_lengths):
    q_offsets = mx.array(np.cumsum([0, *q_lengths]), mx.int32)
    k_offsets = mx.array(np.cumsum([0, *k_lengths]), mx.int32)
    counts = [(length + 31) // 32 for length in q_lengths]
    samples = mx.array(np.repeat(np.arange(len(counts), dtype=np.int32), counts))
    local = mx.array(np.concatenate([np.arange(count, dtype=np.int32) for count in counts]))
    return _tiled(inputs=[q, k, v, q_offsets, k_offsets, samples, local],
                  template=[("Data", q.dtype), ("HeadDim", q.shape[2]), ("Heads", q.shape[1]), ("KVHeads", k.shape[1])],
                  grid=(sum(counts) * 128, q.shape[1], 1), threadgroup=(128, 1, 1),
                  output_shapes=[q.shape], output_dtypes=[q.dtype])[0]

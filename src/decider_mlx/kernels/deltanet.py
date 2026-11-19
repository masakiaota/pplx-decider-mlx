"""Gated DeltaNet kernels for packed sequences: the short causal convolution (+ SiLU) and the delta rule.

The delta rule runs LANES=8 threads per value column (delta_lanes.metal). Each sequence's context is run once and its
state is forked for every question that follows it (isolated questions share the context).
"""
from pathlib import Path

import mlx.core as mx

LANES = 8
_NAMES = ["q", "k", "v", "gamma", "beta", "starts", "context_ends", "question_offsets", "question_starts", "question_ends"]
_delta = mx.fast.metal_kernel(name="decider_delta_lanes", input_names=_NAMES, output_names=["out"],
                              source=(Path(__file__).parent / "delta_lanes.metal").read_text())


def delta(q, k, v, gamma, beta, starts, context_ends, question_offsets, question_starts, question_ends, *, threadgroup):
    """q, k: (T, HK, 128) float32; v: (T, HV, DV) float32; gamma, beta: (T, HV) float32. threadgroup: (32, n, 1);
    the number of simdgroups per threadgroup does not change the values."""
    args = [q, k, v, gamma, beta, starts, context_ends, question_offsets, question_starts, question_ends]
    columns = 32 // LANES
    if not (q.ndim == k.ndim == v.ndim == 3 and q.shape == k.shape and q.shape[2] == 128 and v.shape[2] % columns == 0
            and v.shape[1] % q.shape[1] == 0 and gamma.shape == beta.shape == v.shape[:2]
            and all(a.dtype == mx.float32 for a in args[:5])):
        raise ValueError("Unexpected DeltaNet shapes or dtypes")
    return _delta(inputs=args, template=[("HK", q.shape[1]), ("HV", v.shape[1]), ("DV", v.shape[2]), ("LANES", LANES)],
                  grid=(32, starts.size * v.shape[1] * (v.shape[2] // columns), 1), threadgroup=threadgroup,
                  output_shapes=[v.shape], output_dtypes=[mx.float32])[0]


_conv = mx.fast.metal_kernel(
    name="decider_segment_conv_dyn", input_names=["x", "weight", "starts", "owners", "blocks", "prefixes"], output_names=["out"],
    source=r"""
    const uint index = thread_position_in_grid.x;
    if (index >= uint(x_shape[0]) * C) return;
    const uint token = index / C;
    const uint channel = index % C;
    const int start = starts[owners[token]];
    float result = 0.0f;
    for (int tap = 0; tap < K; ++tap) {
        int position = int(token) - K + 1 + tap;
        int block = blocks[token];
        if (position < block) position = prefixes[token] - (block - position);
        if (position >= start)
            result += float(x[position * C + channel]) * float(weight[channel * K + tap]);
    }
    // Match the explicit BF16 convolution rounding before the activation.
    auto rounded = X(result);
    float value = float(rounded);
    out[index] = X(value / (1.0f + metal::exp(-value)));
    """,
)


def convolution(x, weight, starts, owners, blocks, prefixes):
    size = x.size
    return _conv(inputs=[x, weight, starts, owners, blocks, prefixes],
                 template=[("C", x.shape[1]), ("K", weight.shape[-1]), ("X", x.dtype)],
                 grid=(size, 1, 1), threadgroup=(256, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype])[0]

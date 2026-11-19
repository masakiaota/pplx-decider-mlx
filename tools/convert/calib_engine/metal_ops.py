"""Boundary-aware packed Metal experiments. No dense token-by-token mask."""
import mlx.core as mx


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


_delta = mx.fast.metal_kernel(
    name="decider_segment_delta", input_names=["q", "k", "v", "gamma", "beta", "starts", "context_ends", "question_offsets", "question_starts", "question_ends"], output_names=["out"],
    source=r"""
    const uint lane = thread_position_in_grid.x;
    const uint group = thread_position_in_grid.y;
    const uint first_column = (group % (DV / VB)) * VB;
    const uint value_head = (group / (DV / VB)) % HV;
    const uint sample = group / ((DV / VB) * HV);
    const uint key_head = value_head / (HV / HK);
    float state[VB][DK / 32];
    for (uint j = 0; j < VB; ++j)
        for (uint i = 0; i < DK / 32; ++i) state[j][i] = 0.0f;
    float base_state[VB][DK / 32];
    for (uint phase = question_offsets[sample]; phase <= uint(question_offsets[sample + 1]); ++phase) {
        bool context = phase == uint(question_offsets[sample]);
        uint begin = context ? starts[sample] : question_starts[phase - 1];
        uint end = context ? context_ends[sample] : question_ends[phase - 1];
        if (!context)
            for (uint j = 0; j < VB; ++j)
                for (uint i = 0; i < DK / 32; ++i) state[j][i] = base_state[j][i];
    for (uint token = begin; token < end; ++token) {
        float decay = gamma[token * HV + value_head];
        float rate = beta[token * HV + value_head];
        float keys[DK / 32];
        float queries[DK / 32];
        for (uint i = 0; i < DK / 32; ++i) {
            keys[i] = float(k[(token * HK + key_head) * DK + lane + 32 * i]);
            queries[i] = float(q[(token * HK + key_head) * DK + lane + 32 * i]);
        }
        for (uint j = 0; j < VB; ++j) {
            float predicted = 0.0f;
            for (uint i = 0; i < DK / 32; ++i) {
                state[j][i] *= decay;
                predicted += state[j][i] * keys[i];
            }
            predicted = metal::simd_sum(predicted);
            float difference = (float(v[(token * HV + value_head) * DV + first_column + j]) - predicted) * rate;
            float result = 0.0f;
            for (uint i = 0; i < DK / 32; ++i) {
                state[j][i] += keys[i] * difference;
                result += state[j][i] * queries[i];
            }
            result = metal::simd_sum(result);
            if (lane == 0) out[(token * HV + value_head) * DV + first_column + j] = OUT(result);
        }
    }
        if (context)
            for (uint j = 0; j < VB; ++j)
                for (uint i = 0; i < DK / 32; ++i) base_state[j][i] = state[j][i];
    }
    """,
)


def delta(q, k, v, gamma, beta, starts, context_ends, question_offsets, question_starts, question_ends):
    return _delta(inputs=[q, k, v, gamma, beta, starts, context_ends, question_offsets, question_starts, question_ends],
                  template=[("HK", q.shape[1]), ("HV", v.shape[1]), ("DK", q.shape[2]), ("DV", v.shape[2]), ("VB", 4), ("OUT", v.dtype)],
                  grid=(32, starts.shape[0] * v.shape[1] * (v.shape[2] // 4), 1), threadgroup=(32, 1, 1),
                  output_shapes=[v.shape], output_dtypes=[v.dtype])[0]


_conv = mx.fast.metal_kernel(
    name="decider_segment_conv", input_names=["x", "weight", "starts", "owners", "blocks", "prefixes"], output_names=["out"],
    source=r"""
    const uint index = thread_position_in_grid.x;
    if (index >= N * C) return;
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
                 template=[("N", x.shape[0]), ("C", x.shape[1]), ("K", weight.shape[-1]), ("X", x.dtype)],
                 grid=(size, 1, 1), threadgroup=(256, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype])[0]

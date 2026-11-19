// Derived from experiments/27b-lowmem/003-next/delta8.metal.
// Each group of LANES threads produces one value column; each thread holds four
// bands of 32/LANES adjacent keys at the original 32-element stride.
// Change from the 27B copy: the "single question directly after context" fast
// path is decided per sample at run time instead of by a whole-batch template.
constexpr uint columns = 32 / LANES;
constexpr uint packets = columns / 4;
const uint lane = thread_position_in_grid.x;
const uint sub = lane % LANES;
const uint group = thread_position_in_grid.y;
const uint column = (group % (DV / columns)) * columns + lane / LANES;
const uint head = (group / (DV / columns)) % HV;
const uint sample = group / ((DV / columns) * HV);
const uint key_head = head / (HV / HK);
float4 state[4][packets] = {};
float4 checkpoint[4][packets];
const uint first = question_offsets[sample], last = question_offsets[sample + 1];
const bool one = last == first + 1 && uint(question_starts[first]) == uint(context_ends[sample]);
for (uint phase = first; phase <= (one ? first : last); ++phase) {
    bool context = phase == first;
    uint begin = context ? starts[sample] : question_starts[phase - 1];
    uint end = one ? question_ends[first] :
               context ? context_ends[sample] : question_ends[phase - 1];
    if (!one && !context) {
        for (uint b = 0; b < 4; ++b)
            for (uint p = 0; p < packets; ++p) state[b][p] = checkpoint[b][p];
    }
    for (uint token = begin; token < end; ++token) {
        uint offset = (token * HK + key_head) * 128 + sub * columns;
        uint value_index = (token * HV + head) * DV + column;
        float decay = gamma[token * HV + head];
        float rate = beta[token * HV + head];
        float4 keys[4][packets], queries[4][packets], predicted[packets] = {};
        for (uint b = 0; b < 4; ++b) {
            for (uint p = 0; p < packets; ++p) {
                keys[b][p] = *reinterpret_cast<device const float4*>(k + offset + b * 32 + p * 4);
                queries[b][p] = *reinterpret_cast<device const float4*>(q + offset + b * 32 + p * 4);
                state[b][p] *= decay;
                predicted[p] += state[b][p] * keys[b][p];
            }
        }
        float prediction = 0.f;
        for (uint p = 0; p < packets; ++p) {
            float4 part = predicted[p];
            prediction += (part.x + part.y) + (part.z + part.w);
        }
        for (uint mask = 1; mask < LANES; mask *= 2)
            prediction += metal::simd_shuffle_xor(prediction, mask);
        float difference = (v[value_index] - prediction) * rate;
        float4 result[packets] = {};
        for (uint b = 0; b < 4; ++b) {
            for (uint p = 0; p < packets; ++p) {
                state[b][p] += keys[b][p] * difference;
                result[p] += state[b][p] * queries[b][p];
            }
        }
        float sum = 0.f;
        for (uint p = 0; p < packets; ++p) {
            float4 part = result[p];
            sum += (part.x + part.y) + (part.z + part.w);
        }
        for (uint mask = 1; mask < LANES; mask *= 2)
            sum += metal::simd_shuffle_xor(sum, mask);
        if (sub == 0) out[value_index] = sum;
    }
    if (!one && context) {
        for (uint b = 0; b < 4; ++b)
            for (uint p = 0; p < packets; ++p) checkpoint[b][p] = state[b][p];
    }
}

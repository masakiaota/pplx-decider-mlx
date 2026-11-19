# Kernels

All custom kernels are Metal kernels registered through `mx.fast.metal_kernel`; integer matmuls use the M5 GPU's tensor-op matmul (MPP `matmul2d`). MLX itself is not modified.

## Roofline on M5 Ultra (`E-001`)

Each kernel was timed alone on large square shapes and on the models' real layer shapes (M = 427 to 5,368 rows).

| Operation | Measured peak | Note |
|---|---:|---|
| BF16 matmul | about 100 TFLOP/s | Matches the published per-core figure scaled to 64 cores; this is the hardware ceiling |
| INT8 x INT8 | 165-177 TOP/s | 90-96% of the scaled published figure (184) |
| u8 x u8 (Flash shared affine8) | 150-167 TOP/s | Essentially at the INT8 ceiling |
| int8 x int4 (s4) | 150-161 TOP/s | Matches the scaled published figure (151) |
| u8 x uint4 (first 27B W4A8) | 140-150 TOP/s | 5-10% below s4 |
| Standard MLX quantized matmul, 4-bit, 8-bit, MXFP4 | about 100 | Dequantizes to BF16, so capped at the BF16 ceiling |

Consequences:

- Weight-only quantization saves memory and bandwidth but cannot exceed BF16 compute. Faster prefill requires integer activations.
- Flash's INT8 matmul is already at 90-95% of its ceiling as a kernel. Inside the model the linear layers reach 113-154 TOP/s (65-90% of the ceiling), so the remaining headroom is in small-M efficiency and in everything that is not a matmul.
- Activation quantization costs about 2-3% of matmul time.

**What is left if every linear layer ran at its ceiling** (from per-category knock-out timing, `E-002`): Flash single -18%, batch-20 -2%, long 5K -8%, Locate -43%; 27B W4A8 single -20%, batch-20 -8%, long 5,368 -16%, Locate -40%. Part of this was then recovered by s4 (27B) and NAX attention (Flash, not shipped for short inputs). Locate stays far from the ceiling in both models because it does not use the integer path.

The knock-out method replaces all kernels of one category with zeros of the same shape (kept in the graph with `mx.depends`) and measures the drop in wall time. A first version without the dependency let MLX drop whole subgraphs, so categories summed to 130%; this was fixed before the numbers above.

## INT8 (Flash)

Activations are quantized per row to INT8 (round, clamp) and multiplied with 8-bit weights stored as unsigned codes plus a per-row scale; the epilogue corrects for the unsigned offset and applies both scales. The same weight bytes are read by MLX as group-64 affine 8-bit, so there is one copy of the weights. The best tile for INT8 is 128x64 or similar. The matmul runs on all rows in one call; the earlier 2,048-row blocking gave identical outputs and was 7-15% slower on long inputs.

## W4A8 and s4 (27B)

**Diagnosis (`E-003`).** The first W4A8 kernel (uint8 activations x uint4 weights) was 13-23% slower than INT8 x INT8. Breaking down the gap:

- The correction epilogue was not the cause (within 2%).
- The unsigned 4-bit path itself is 5-10% slower than the signed one at the same tile.
- The tile rule was wrong for 27B `down` (K > 8,192): 128x64 was chosen, while 32x64 with two simdgroups is best. `down` went from 120 to 149 TOP/s (+24%, INT8 level).

**s4 (`E-004`, adopted).** Unsigned 4-bit computes sum((a+128)(q+8)) and subtracts correction terms; signed int8 x int4 computes sum(a q) directly. Both give the same int32 result, so outputs are bit-exact. The only change to the weights is XOR 0x88888888 on each packed 32-bit word (offset-8 nibbles to two's complement); row scales are unchanged and row sums are no longer needed. Tiles come from a per-(N, K) table measured on the real shapes. Whole-model times relative to the unsigned kernel: single 0.91, list-20 0.92, long 0.93 (0.87-0.97 across 20 feature cases). With small tiles W4A8 reaches 0.94-1.01x of INT8 x INT8 at large M, and 0.84-0.94x at M = 427.

Recoding at load time doubles the anonymous memory (+12 GB), so the shipped weights are stored already recoded, with a metadata field stating the nibble encoding. Files in the old encoding are refused.

**W4A8 on Flash (rejected).** On Flash's smaller shapes, W4A8 was 11-38% slower than u8 x u8 and 18-25% slower after a tile search (measured before s4). Flash keeps 8-bit weights.

## Runtime-M kernels (`A-002`, `B-003`)

The INT8 activation quantizer, the integer matmuls and the short convolution before DeltaNet took the row count as a Metal template constant, so every new input length compiled new pipelines (0.1-0.9 s per new M on 27B shapes; about 230 ms per new length on Flash). The row count is now read from a small buffer or the input shape at run time. Outputs are bit-exact on 24 shape/M combinations. Effect on Flash: first request at a new length 265-296 ms -> 41-60 ms. On 27B, together with a short warm-up, it brought the first request of a process from 4.84 s to 0.64-0.80 s.

## DeltaNet recurrence

**8 lanes per value column (adopted).** The original kernel had each 32-lane simdgroup walk four value columns in sequence. The 8-lane kernel splits the simdgroup so each value column is computed by 8 lanes in parallel, keeping the original summation order (32-element strided partial sums along the key dimension) so FP32 outputs are bit-identical. The idea follows the mlx-lm implementation. On 27B it saved 0.5-0.9%; on Flash, where DeltaNet is a larger share, 2-5% (single -4.7%, batch-20 -2.1%, long 5K -4.1%) (`27b-lowmem/001`, `flash-speed/001`). The Flash version also fixed a fast-path condition that was decided per batch and could misfire when samples with zero and two questions were mixed; it is now decided per sample.

**Threadgroup (32,4,1) (adopted, `L-007`).** Grouping four column groups per threadgroup instead of one: bit-exact, single -2.3 to -2.6%, list-20 -1.3%, bulk -0.5 to -0.9%, long -0.3 to -0.7%.

**What remains.** In Flash, DeltaNet layers are 37-42% of time: projections 16-20%, recurrence 9-15%, q/k normalization and casts 3-6%, gated norm 1-4%, convolution 1-3%, gates about 0%. No further bit-exact change was found. The DeltaNet projections of long single inputs could move to INT8 (estimated about -10% on long 5K); not done.

## Attention

Short inputs (up to 512 tokens per chunk) use a simple segment attention kernel (one simdgroup per query and head). Longer inputs use a tiled variable-length kernel built on MLX's NAX "steel" attention code, at about 65 TFLOP/s for 5,368 tokens.

**NAX attention for short inputs (rejected, `E-005`, `A-009`).** Attention was about 10% of a Flash single request. Using the tiled kernel for all lengths: single 0.91x, batch-20 0.95x. It changes numbers (different softmax and accumulation order): gate 100/100 with zero flips, but max |dp| 0.12-0.16, Locate span agreement 98/100, and, with Locate excluded, 298/300 vs 300/300 binary agreement. Not adopted; the kernel is used only above 512 tokens, as before.

## Other kernel-level items

- Matmul boundaries: MLX's standard quantized matmul and BF16 matmuls switch kernels at certain row counts, which changes the bits of every row. The full list was measured for M = 1-2,304 on MLX 0.32.5 / M5 and is shipped as a data file; length buckets never cross these boundaries (see [serving.md](serving.md)). INT8 matmul, RMSNorm and the BF16 `in_proj_b` are row-independent. The file must be remeasured when MLX changes.
- Command-buffer tuning, fused gate/up, low-rank fused preparation, transposed q4 storage and q8 stage splitting were all slower or within noise in earlier rounds and were dropped.

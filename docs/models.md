# Models

Both models share the code; everything that differs between them is in `src/decider_mlx/registry.py` (one
`ModelSpec` each) and `src/decider_mlx/data/boundaries-m5.json` (length buckets and kernel edges measured on M5).
The research behind each choice is in [experiments/](../experiments/README.md).

| | `pplx-decider-flash-v1.3` | `pplx-decider-v1.3` |
|---|---|---|
| Original | [perplexity-ai/pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3), on Qwen3.5-4B | [perplexity-ai/pplx-decider-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-v1.3), on Qwen3.8-27B |
| Layers | 32 (24 Gated DeltaNet, 8 full attention) + vision tower | 64 (48 Gated DeltaNet, 16 full attention) + vision tower |
| Weights repository | [masakiaota/pplx-decider-flash-v1.3-mlx](https://huggingface.co/masakiaota/pplx-decider-flash-v1.3-mlx) | [masakiaota/pplx-decider-v1.3-mlx](https://huggingface.co/masakiaota/pplx-decider-v1.3-mlx) |
| Download | 9.6 GB (main 5.74 GB, Locate 3.80 GB) | 16.7 GB (including the Locate LoRA) |
| Linear layers | shared affine 8-bit; INT8 x INT8 matmul for the MLP and input projections | row-wise symmetric 4-bit (s4); W4A8 matmul for all linear layers except k/v |
| Score questions | standard 8-bit matmul in layers 0-15, INT8 in 16-31 | the same W4A8 path |
| Locate | LoRA merged into separate 8-bit weights, loaded on the first Locate request | BF16 LoRA (rank 64) on top of the 4-bit weights, loaded on the first Locate request |
| Saved temperature | 1.3302962 | 1.2263882 |
| Recommended memory | 24 GB or more | 36 GB or more |

## Flash

**Weights.** One 8-bit file serves two matmuls. The codes are stored as unsigned 8-bit with one scale per output
row and are described to MLX as group-64 affine 8-bit (bias = -128 x scale), so MLX's quantized matmul and the INT8
matmul read the same bytes. This removed a second INT8 copy (8.84 -> 5.74 GB mapped) with no quality change.
Embeddings, norms, convolutions, the vision tower and the readout head stay BF16.

**Arithmetic.** Whenever a forward has at least 128 rows, activations of the MLP and input projections are
quantized per row to INT8 and multiplied with the 8-bit codes. INT8 roughly quadruples the error of `score`
distributions against BF16, and that error comes from early layers, so a sequence that contains a score question
uses the standard 8-bit matmul in layers 0-15. The arithmetic is chosen per sequence, so a binary answer does not
change when a score question is in the same request. Locate runs on the standard 8-bit path: INT8 made span
agreement significantly worse.

**Kernels and graphs.** 8-lane DeltaNet recurrence with (32, 4, 1) threadgroups; NAX tiled attention for inputs over
512 tokens; runtime-M kernels, so a new length does not rebuild a pipeline. Short text and image requests run on
compiled graphs per length bucket (up to 7 sequences per forward), text lists of decision questions on a
16-sequence graph, single sequences up to 32,768 tokens on per-shape compiled graphs, and the 24 vision blocks as one
compiled function. Arrays of 64 or more requests are packed into fixed 2,048-token batches.

**Startup.** The main weights are memory-mapped and read once by a background thread after loading. The
preparation then traces (without running) the bucket and image graphs and runs 36 short synthetic warm-up requests,
between real requests.

**Quality** against the original BF16 model (held-out sets; method in
[experiments/quality-method.md](../experiments/quality-method.md)):

| | Same answer as BF16 |
|---|---:|
| Binary (`noul`) | 821/826 |
| `choice` | 222/224 |
| `score` argmax | 73/74 |
| Locate span sets | 305/314 |
| Long inputs 12K-32K tokens | 9/9 |
| BoolQ gate, separate / isolated, single / list | 100/100 each |

Mean |dp| on binary questions is about 0.009. Using the standard 8-bit matmul everywhere would cut that error by about
60% but make requests 16-22% slower; speed was preferred.

**Performance** (Mac Studio M5 Ultra, release package, HTTP unless noted):

| Case | Time |
|---|---:|
| Process start to accepting requests | 0.44-0.47 s |
| Process start to the first answer (427 tokens) | 0.51-0.54 s |
| `load()` returns (Python) | 0.30 s |
| Preparation done, no requests arriving | 5.4 s |
| One request, 427 tokens, repeated | 43 ms |
| 20 requests in one array | 0.31 s |
| 520 mixed requests, one at a time | 26.5 s (median 32 ms) |
| Long 5.4K / 12.9K / 16.9K / 24.3K / 32.4K tokens | 0.42 / 2.17 / 3.12 / 4.75 / 7.03 s |
| 1,072 mixed requests in arrays (1,026 separate + 46 isolated) | 45.2 + 1.8 s |

Memory: 5.74 GB mapped main weights (+3.80 GB after the first Locate request). After the HTTP benchmark, MLX active
9.56 GB, peak 14.17 GB, RSS 7.2 GB. A 32.4K-token request peaks at 11.2 GB without Locate and 14.2 GB for a 32.4K
Locate request.

## 27B

**Weights.** All linear layers except the k/v projections are row-wise symmetric 4-bit: one FP32 scale per output
row, codes stored as two's-complement int4 ("s4") so the hardware's int8 x int4 path computes the integer sums
directly. Rounding uses GPTQ-style error compensation with act-order and a per-row scale search, calibrated on about
240K tokens (BoolQ, SQuAD v2 and Qasper training splits and SQuAD Locate requests) with no overlap with any
evaluation set. Each of the 368 row-wise tensors is its own file under `rowq4/`. The k/v projections stay group-64
affine 4-bit; embeddings, norms, convolutions, the DeltaNet gate projections, the vision tower and the readout head
stay BF16.

**Arithmetic.** INT8 activations x s4 weights (W4A8) for every feature, with the scales applied once in the
epilogue and tile sizes from a per-shape table (`data/tile-table.json`). Locate adds the BF16 LoRA on top of the
same 4-bit weights, so only one set of weights is mapped.

**Kernels and graphs.** 8-lane DeltaNet recurrence; length buckets with compiled graphs for inputs up to 512
tokens (up to 8 sequences per forward); single text requests over 4,096 tokens on per-shape compiled graphs (below
that, eager is faster on fresh lengths); bulk packing for arrays of 64 or more. Images run eagerly.

**Startup.** The weights are memory-mapped; the preparation runs 5 synthetic warm-up requests (text only) and traces
the bucket graphs between real requests.

**Quality** against the original BF16 model:

| | Same answer as BF16 |
|---|---:|
| Binary (`noul`) | 717/726 |
| `choice` | 161/164 |
| `score` argmax | 70/74 |
| Locate span sets (964-question harness) | 100/114 |
| Locate span sets (300-question set) | 267/300 |
| BoolQ gate, separate / isolated, single / list | 100/100 each |

Mean |dp| on binary questions is 0.019, above the earlier group-64 4-bit build (0.016) with equal decision
agreement; the difference comes from the row-wise 4-bit weights. Locate agreement with gold labels is at the original
model's level (F1 0.697 vs 0.685).

**Performance** (Mac Studio M5 Ultra, release package, HTTP unless noted):

| Case | Time |
|---|---:|
| Process start to accepting requests | 0.3-0.5 s |
| Process start to the first answer (427 tokens) | 0.6-1.0 s |
| `load()` returns (Python) | 0.28 s |
| Preparation done, no requests arriving | 7.4 s |
| One request, 427 tokens, repeated | 220 ms |
| 20 requests in one array | 1.92 s |
| 520 mixed requests, one at a time | 116.7-116.8 s (median 135 ms) |
| Long 5.4K / 12.9K / 16.9K / 24.3K / 32.4K tokens | 2.66 / 7.62 / 10.65 / 16.30 / 22.86 s |
| 1,072 mixed requests in arrays (1,026 separate + 46 isolated) | 227.3 + 8.9 s |

Memory: 16.67 GB mapped including Locate. MLX peak 23.72 GB after the HTTP benchmark and 25.0 GB for a 32K-token
Locate request; RSS about 1.7 GB, because the weights are memory-mapped. The original BF16 model peaks at 57.1 GB.

## Known differences and limits

- Long inputs are 4-9% (Flash) and 4-6% (27B) slower in the release package than in the research engines with the
  same graphs and identical outputs; the cause is not yet found.
- On 27B, image outputs depend on the first image a process sees when that image is synthetic; the warm-up
  therefore has no image step. The cause is not found.
- Outputs are bit-exact on the M5 Ultra they were recorded on (`tests/golden`); other M5 chips, macOS versions or
  MLX builds may differ in the last bits.

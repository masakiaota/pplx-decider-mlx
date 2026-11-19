# Research record

This directory explains how the two engines in this repository were built: which techniques were tried, which were kept, which were dropped, and why. It is text only. Weights, raw logs and full result tables stay in the private research workspace; experiment IDs in parentheses (for example `G-001`) refer to that workspace and are given only so that statements can be traced.

All measurements were taken on one Mac Studio with an M5 Ultra (64-core GPU) using MLX 0.32.5, one GPU job at a time, with a 64 GB cap on total process memory and no swap growth allowed. Numbers on other Macs will differ; ratios are more portable than absolute times.

## Starting point

The goal was a Mac engine for Perplexity Decider v1.3 in two sizes, Flash (about 4B parameters, 32 layers, 24 of them Gated DeltaNet) and 27B (64 layers), that is faster than the official Mac app and stays as close as possible to the original BF16 model.

Earlier rounds produced two working baselines:

- Flash: one 8-bit affine weight file that serves both the standard MLX quantized matmul and a custom INT8 x INT8 matmul ("shared affine8", 5.74 GB of main weights).
- 27B: group-64 4-bit weights with GPTQ-style rounding compensation plus compiled long-input graphs ("B015", 17.2 GB of main weights).

Two later rounds (one overnight, one daytime) turned these into the shipped defaults. The overnight round brought every feature onto the fast path, built the quality harness, recalibrated the 27B weights and measured the hardware ceiling. The daytime round audited both engines against the official API, fixed what differed, and reworked startup, bucketing and bulk processing.

## How quality was measured

Quality means agreement with the original model, not accuracy on labels. Every candidate is compared against BF16 reference outputs produced by an MLX port of the original model, which reproduces the official PyTorch outputs (27B: 100/100 decisions on the gate set, mean |dp| 0.0014). The harness reports decision agreement, mean |dp|, KL, flips near the threshold, Locate span agreement and paired bootstrap confidence intervals between candidates. A 100-question BoolQ gate (at least 96/100 under four execution conditions) is a smoke test; decisions about adoption use the 1,000+ question sets and paired continuous metrics, because the gate alone cannot separate candidates. Details are in [quality-method.md](quality-method.md).

"Bit-exact" below means the engine returns byte-identical outputs to the previous version on the same inputs. Most speed changes were required to be bit-exact; changes that alter numbers had to show a non-significant paired difference against the previous default.

## Index

| File | Contents |
|---|---|
| [flash.md](flash.md) | Flash engine: weights, INT8 arithmetic, score layers, Locate, images, long inputs, audit fixes, final numbers |
| [27b.md](27b.md) | 27B engine: row-wise 4-bit weights, W4A8, recalibration, Locate, final numbers |
| [kernels.md](kernels.md) | INT8, W4A8 and s4 matmuls, roofline measurements, DeltaNet lanes, NAX attention, runtime-M kernels |
| [serving.md](serving.md) | Length buckets, bulk packing, startup and trace-only preparation, graph export, cross-process lock, in-process overlap, memory limits |
| [quality-method.md](quality-method.md) | Evaluation sets, BF16 references, harness, gate, calibration/evaluation de-duplication, statistical resolution |

## Final results in one paragraph

Flash: about 2x the official Mac app (v1.2 Flash, q8_0 GGUF) in steady state on the same HTTP workload (single 427-token request 88.7 ms vs 44.8 ms; random-length stream of 420 requests 1.92x). Against the original Flash BF16 it agrees on 821/826 binary questions, 222/224 choice, 73/74 score argmax, 305/314 Locate span sets, and 9/9 on 12K-32K token inputs. Main weights 5.74 GB, plus a 3.80 GB Locate weight file loaded only on first Locate use. The official app is a different model version (v1.2), so the speed ratio compares products, not identical models. Times measured on the release package are in [docs/models.md](../docs/models.md); they can differ by a few percent from the research-engine times quoted in this directory.

27B: against the earlier 4-bit default (B015) the final engine is 0.61-0.82x the time on 20 feature cases, and 0.58-0.61x on 12K-32K inputs. Agreement with the original 27B BF16 is statistically equivalent to the previous candidate and to B015 on decisions; Locate span agreement is lower than B015 (267/300 vs 280/300) and was accepted. Mapped weights 16.67 GB including Locate; MLX peak 25.0 GB at 32K tokens (original BF16: 57.1 GB peak).

## Adopted and rejected techniques

| Technique | Model | Decision | Effect | Reason |
|---|---|---|---|---|
| Shared affine8 weights (one 8-bit file for standard and INT8 paths) | Flash | Adopted | Weights 8.84 -> 5.74 GB mapped, speed within +-2% | Removes duplicate INT8 copies with no quality loss |
| INT8 activations for MLP and input projections | Flash | Adopted | Main speed source; u8 x u8 matmul runs at 90-95% of INT8 peak | Agreement with BF16 equal to standard 8-bit within CI |
| Score questions use standard arithmetic in layers 0-15 | Flash | Adopted | Score error 0.060 -> 0.017, keeps 1.17-1.26x of the INT8 gain | INT8 error on score comes from early layers |
| Standard arithmetic for all question types | Flash | Rejected | Error -60%, but 1.16-1.22x slower | Speed preferred |
| Row-wise 4-bit weights with W4A8 matmul | 27B | Adopted | 0.83-0.94x of B015 time, 1.5 GB less weight | Needs recalibration to match quality (below) |
| W4A8 on group-64 weights | 27B | Rejected | Slower than standard QMM | Per-group correction too expensive |
| Recalibration with 240K non-overlapping tokens, act-order, row-scale search | 27B | Adopted | Layer error about halved; decisions back to B015 level | Earlier calibration was 1K tokens taken from the eval set |
| s4 kernel (signed int4 weights, small tiles) | 27B | Adopted | 0.87-0.97x, bit-exact | Unsigned path and large tiles were the bottleneck |
| 8-lane DeltaNet recurrence | Both | Adopted | Flash -2 to -5%, 27B about -1%, bit-exact | Parallelizes value columns |
| DeltaNet threadgroup (32,4,1) | Flash | Adopted | -0.3 to -2.6%, bit-exact | Last bit-exact DeltaNet gain found |
| Runtime-M kernels | Both | Adopted | First request at a new length 265-296 -> 41-60 ms (Flash) | No pipeline rebuild per length |
| NAX tiled attention for short inputs | Flash | Rejected | -5 to -9% short paths | Locate 98/100, two extra flips |
| Length buckets with precompiled graphs | Both | Adopted | Random-length stream -17% (Flash), bit-exact | Removes per-length trace cost |
| Bulk packing into fixed 2048-token packs (64+ requests) | Both | Adopted | Flash 1,072 mixed jobs 64.8 -> 47 s; 27B 0.71x | Output independent of packing; quality equal |
| Locate merged sidecar (LoRA folded into weights), lazy load | Flash | Adopted | Locate -5 to -20% | +3.8 GB only when Locate is used |
| Locate in INT8 | Flash | Rejected | Faster | Significantly worse agreement |
| Locate W4A16 or extra B015 projections | 27B | Rejected | Span agreement 274-277/300 | 12-42% slower Locate or +2.83 GB |
| Vision tower INT8 | Flash | Rejected | -7 to -8% | Agreement drops (vqa60 choice 58/60) |
| Image bucket graphs and vision-tower block compile | Flash | Adopted | vqa -11 to -13%, bit-exact | Same mechanism as text buckets |
| Compile of two-chunk large images | Flash | Rejected | -5% | Significant |dp| increase |
| Long-input compile up to 32,768 tokens | Both | Adopted | 12.9K-32.4K: 0.58-0.71x | Official max length; compile beats eager above about 3-4K tokens |
| Listen first, prepare between requests | Both | Adopted | First answer 10.8-14.2 s -> 3.2 s (Flash), 8.3 -> 3.3 s (27B) | Bit-exact regardless of timing |
| Trace-only preparation (no dummy execution) | Both | Adopted | Preparation 12.1 -> 7.1 s (Flash), 22.9 -> 9.3 s (27B); 5.4 / 7.4 s in the release package | Execution, not tracing, dominated |
| Graph export and reload | Both | Rejected | No startup gain | Import is slower than tracing |
| Cross-process GPU lock | Both | Rejected | 11-35% slower than no lock | Concurrent processes fill GPU gaps |
| Two threads in one process | Both | Deferred | -14% | 10/120 outputs differ; engine state is single-threaded |
| Fixed engine memory limit and preflight | Both | Removed | — | Left to MLX defaults |

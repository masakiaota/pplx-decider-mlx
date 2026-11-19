# Flash (Decider v1.3 Flash)

Flash has 32 layers: 24 Gated DeltaNet layers and 8 full-attention layers, plus a vision tower. At short lengths about 55-70% of time is spent in linear layers, about 37-42% in DeltaNet layers (including their projections), and about 10% in attention (`E-002`, `L-007`).

## Earlier history (brief)

- An MLX port of the model was built first, then 8-bit weights via memory-mapped safetensors. A plain 8-bit mmap engine already beat the official Mac app (v1.2 q8_0) by about 2x on a 20-request batch.
- Row-wise INT8 activations for the MLP and input projections gave the next large gain (single request -14%), but needed a second INT8 copy of those weights (+3.1 GB).
- "Shared affine8" removed that copy: the INT8 codes are stored as unsigned 8-bit (q+128) with a per-row scale, and the same bytes are described to MLX as group-64 affine 8-bit with bias = -128 * scale. Standard quantized matmul and the custom INT8 matmul read the same array. Mapped weights went from 8.84 to 5.74 GB with speed within +-2% and 100/100 on the gate (`flash-shared-q8`). This file is the base of the shipped engine.
- 4-bit Flash variants (shared q4, W4A8 on Flash shapes) were 11-38% slower than shared affine8 and were dropped. CPU-side fixed cost per request was measured at about 2.4 ms of 45 ms, so rewriting the host side was not pursued (`flash-speed/002`).

## Arithmetic

**INT8 MLP and input projections.** Activations are quantized per row to INT8 and multiplied with the shared 8-bit weights by an INT8 x INT8 matmul. Applied whenever a forward has at least 128 rows, regardless of length or batch. The audit removed an unfounded upper bound (2,048 rows) and the 2,048-row split; long inputs got 7.5-15% faster with identical outputs (`I-001`).

**Score layer split.** Score questions read a probability distribution over levels, and INT8 roughly quadrupled their error against BF16 (mean |dscore| 0.0146 -> 0.060). Layer sweeps on calibration data showed the error is introduced in early layers. Sequences that contain a score question now use standard 8-bit arithmetic in layers 0-15 and INT8 in 16-31: error 0.0169, argmax 59/60 (same as standard), and score batches 1.17x and long score inputs 1.26x faster than full standard arithmetic (`A-008`). Arithmetic is chosen per sequence, so a binary question's answer no longer changes when a score question is in the same request (the audit found p 0.893 -> 0.788 before this fix). The cost is that choice + score in one request is about 10% slower, and the choice part there is INT8 (mean |dp| 0.0059 -> 0.0201 on that subset).

**Standard arithmetic everywhere (rejected).** Using the score arithmetic for every question type cuts BF16 error by about 60% (binary mean |dp| 0.0094 -> 0.0035) but makes single requests 42 -> 49 ms and batches 1.16-1.22x slower. Rejected in favour of speed.

**Raw logits.** The raw path also runs INT8 and compiled; the returned logits are now consistent with the probabilities (previously softmax(raw) and the reported probability differed by up to 0.138).

## Locate

Locate (a local extension question type that returns text spans) uses a LoRA adapter on top of the base model.

- **INT8 for Locate (rejected).** Over all layers or selected layers, agreement with BF16 span scores was significantly worse than the standard path, although F1 against gold labels was unchanged (0.720 vs 0.723) (`L-001`).
- **BF16 LoRA everywhere (rejected).** Same quality, but +17% on single requests.
- **hybrid3 LoRA (adopted, later superseded).** A in BF16 for up to 1,024 rows, A and B in BF16 above: Locate -5 to -27%, MLX peak -0.9 GB, no new weights (`L-005`).
- **Merged sidecar (adopted).** BF16(W + B A alpha/r) quantized to affine 8-bit g64, 3.80 GB, used only for Locate; the runtime LoRA disappears. Against the hybrid3 version: Locate single -14%, 12.9K Locate -20%, mixed lists -7 to -9%. Non-Locate outputs are bit-exact. Locate span agreement with BF16 305/314 (hybrid3: 310/314, difference not significant), gold F1 0.704 vs 0.714. The file is loaded on the first Locate request (about 0.4 s), so workloads without Locate do not pay for it; after Locate, MLX active memory is 9.55 GB (`L-002`, `L-003`, `I-011`).

## Images

- **Vision tower INT8 (rejected).** -7 to -8% on fresh images, but agreement dropped (vqa60 choice 58/60) (`A-009`).
- **Image bucket graphs and vision-tower block compile (adopted).** Short forwards that contain images use the same length buckets as text; the 24 vision blocks are compiled as one function. Bit-exact, vqa -11 to -13%, one image with two questions over HTTP 120 -> 105 ms (`V-002`).
- **Large images (open).** Above 512 tokens per chunk, compiling the two-chunk separate path is 5% faster but significantly worse against BF16 (|dp| +0.0017 [+0.0003, +0.0035]), so it was not adopted (`V-003`). A 1.6 MP image with two questions takes about 0.5 s; the vision tower alone is 150-180 ms there.
- Image preprocessing on the CPU is about 2 ms; an earlier "60 ms outside forward" was vision-tower and trace time, not preprocessing (`V-001`).

## Long inputs

The official maximum length is 32,768 tokens; the research engine had a 12,000 limit in three places. Inputs up to 33.6K tokens run and agree with BF16 10/10 (max |dp| 0.031), MLX peak 11.9 GB vs 47.3 GB for BF16 (`A-009`). A single sequence is compiled up to the maximum length; for fresh requests compile is about 15% faster than eager. 12.9K-32.4K inputs take 0.66-0.71x of the eager time (32.4K: about 9.1 -> 6.4 s). Long inputs are not bucketed: the trace is only 0.1-0.2 s per request, and sharing graphs across lengths would need an attention kernel that takes the length as an input (`I-001`).

## Profiles (removed)

Several opt-in profiles were built overnight: `unique` (no compile for short text, -11% for always-new requests but repeats 42 -> 51 ms), `unique_balanced` (-3.4%), `balanced_nax` (NAX attention for short paths, -4 to -10%, two extra flips). The shipped engine has a single default and no profiles; length buckets gave the `unique` gain without its repeat penalty.

## Audit against the official specification

The daytime round audited the engine against the official API, library and model card (`U-001`, `U-004`). Main fixes:

- Max length 12,000/8,192 -> 32,768; over-length requests raise the official error instead of truncating; isolated mode and Locate split questions in two first.
- Default temperature was 1 (left over from BoolQ evaluation) -> the stored temperature (Flash 1.3303).
- HTTP silently treated `image_url` parts in `state` as text -> converted to images; unknown fields and wrong model names return the official 400 error; internal failures return 500.
- A batch-16 compile was chosen based on free memory, so outputs could change under memory pressure -> chosen by structure only.
- Measurement leftovers (guard asserts, profiles, environment switches, debug hooks) removed.

Checked and found correct: question splitting, token counts (100/100 identical to the official tokenizer path), image preprocessing, chat template, Locate threshold.

## Final engine

Shared affine8 weights, INT8 MLP and input projections, score layer split, length buckets for text and images, vision-tower compile, long-input compile up to 32K, bulk packing for 64+ requests, Locate merged sidecar (lazy), 8-lane DeltaNet with (32,4,1) threadgroups, NAX attention for inputs over 512 tokens only.

Against the official Mac app (v1.2 Flash q8_0, same HTTP body, fresh server processes, `U-005`):

| Case | Official v1.2 | This engine | Ratio |
|---|---:|---:|---:|
| Single 427 tokens | 88.7 ms | 44.8 ms | 1.98 |
| 20 requests sequential | 1.211 s | 0.582 s | 2.08 |
| 20 requests, 4 concurrent | 1.192 s | 0.550 s | 2.17 |
| Long 5.4K | 0.956 s | 0.424 s | 2.25 |
| 420 random-length requests, median / p95 | 59.5 / 239 ms | 30.7 / 153 ms | 1.92 (total) |
| Start to first answer | 2.2 s | 3.2 s | official faster |
| Locate, lists | not supported | supported | — |

Quality against the original Flash BF16: gate 100/100 under all four conditions; binary 821/826, choice 222/224, score argmax 73/74, Locate 305/314, long 12K-32K 9/9.

Memory: main weights 5.74 GB mapped; MLX active 5.75 GB before Locate and 9.55 GB after; MLX peak 11.15 GB at 32.4K tokens without Locate and 14.1-14.2 GB for 32.4K Locate after the sidecar is loaded; RSS 7.3-8.2 GB.

## Open items

- Startup to first answer was 0.6-0.9 s slower than the official app in the research engine; most of it was importing transformers and building the processor (about 2 s). The release package replaced both with its own tokenizer, chat template and Metal image preprocessing (byte-identical inputs, `V-004`) and maps the weights without a blocking warm-up: process start to first answer is now 0.51-0.54 s (Flash) and 0.6-1.0 s (27B) (`R-002`).
- DeltaNet layers are 37-42% of time; no further bit-exact improvement was found beyond 0.3-2.6%.
- Linear layers in Locate run at about 61 TFLOP/s against a 165 TOP/s INT8 peak; the INT8 route is closed for quality reasons.

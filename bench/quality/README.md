# Quality against the original model

These tools measure how close a decider-mlx build is to the original Perplexity Decider v1.3 model: they build
held-out evaluation sets from public datasets, produce BF16 reference outputs with the original checkpoint, run a
build over the same sets, and compare the two. They produced the numbers in the main [README](../../README.md#quality)
and [docs/models.md](../../docs/models.md); the method and its reasoning are in
[experiments/quality-method.md](../../experiments/quality-method.md).

No dataset content is in this repository. The scripts download the sources and build the sets locally under
`bench/quality/data/` (git-ignored). Outputs go wherever `--out` points; `bench/quality/out/` is git-ignored too.

| Step | Script | Time (M5 Ultra) | Memory |
|---|---|---|---|
| 1. Download the sources | `download.py` | under a minute, about 45 MB | small |
| 2. Build and check the sets | `build_sets.py`, `check_sets.py` | under a minute, plus about 120 image downloads | under 1 GB |
| 3. BF16 references | `reference.py` | Flash about 6 min, 27B about 20 min (all 1,072 requests) | MLX peak Flash 12 GB, **27B 57 GB** |
| 4. Run a build | `run.py` | Flash 2-3 min, 27B 5-8 min | as in normal use (Flash about 11 GB, 27B about 25 GB) |
| 5. Compare | `compare.py`, `locate_soft.py`, `resolution.py` | seconds | small |

All commands below run from the repository root.

## 1. Download the sources

```sh
uv run python bench/quality/download.py
```

| Dataset | Source (pinned) | Used for | Licence |
|---|---|---|---|
| BoolQ | [google/boolq](https://huggingface.co/datasets/google/boolq) @ `35b264d0`, train and validation Parquet | binary, choice and score questions; the gate | CC BY-SA 3.0 |
| SQuAD v2 | [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) @ `3ffb306f`, train and validation Parquet | choice, multi-question, Locate (with unanswerable questions) | CC BY-SA 4.0 |
| Qasper | [allenai/qasper](https://huggingface.co/datasets/allenai/qasper) Parquet conversion @ `06806e46`, train and validation | long inputs (paper text, yes/no questions) | CC BY 4.0 |
| VQAv2 | [lmms-lab-encoder/VQAv2](https://huggingface.co/datasets/lmms-lab-encoder/VQAv2) @ `32665d35` validation, rows from the dataset viewer API | images (choice and binary) | CC BY 4.0 (annotations); the images are from COCO and keep their Flickr licences |

The Parquet files are checked by sha256. VQAv2 rows come from the
[dataset viewer API](https://huggingface.co/docs/dataset-viewer), which has no pinned revision; the build checks the
resulting files instead (step 2). No Hugging Face account is needed. The built sets and every output file contain
text and images of these datasets and are under their licences; do not commit or redistribute them as part of this
repository.

## 2. Build and check the sets

```sh
uv run --with pyarrow python bench/quality/build_sets.py
uv run python bench/quality/check_sets.py
```

`build_sets.py` needs `pyarrow` to read Parquet; `uv run --with pyarrow` adds it for that one command without
changing the project. It downloads the selected VQAv2 images (the viewer's image URLs expire: if a download fails,
run `download.py --refresh-vqa` and build again) and compares every file with [expected.json](expected.json), so you
get exactly the sets the published numbers were measured on.

Evaluation sets (`data/eval/`), selected with fixed seeds:

| Set | Requests | Content |
|---|---:|---|
| boolq100 | 100 | The gate: BoolQ validation, binary |
| boolq-val300 | 300 | BoolQ validation without the gate, binary |
| boolq-choice-score60 | 60 | BoolQ validation, 3-way choice and 5-level score |
| squad-choice60 | 60 | SQuAD v2 validation, the answer and 3 answers of other questions on the same article |
| squad-locate100 | 100 | SQuAD v2 validation, Locate, 70 answerable and 30 unanswerable |
| squad-multi30-isolated / -separate | 30 + 30 | four "does this passage answer ..." questions per context, in both modes |
| vqa30 | 30 | VQAv2 validation, 4-way choice and binary, distinct images |
| qasper-long30 | 30 | Qasper validation yes/no, paper text cut to about 3K-28K characters (up to 5.9K tokens) |
| features16-separate / -isolated | 16 + 16 | feature probes: mixed question types, object and array states, several questions, images, Locate |
| squad-locate200b | 200 | more Locate, contexts disjoint from the sets above |
| qasper-long40b | 40 | more long inputs (1.7K-7K tokens), papers disjoint from qasper-long30 |
| vqa60b | 60 | more images, disjoint from vqa30 and calibration |

The first eleven files are the core suite (`--sets core`: 772 requests, 1,078 questions); the last three are
`--sets extra` (300 requests). Each line is `{"id", "set", "request", "options", "label", "meta"}`: the request as
sent to the engine, its options (execution mode, temperature 1), the gold label and the source row. Image paths are
relative to the file's folder.

The calibration pool (`data/calib/`: BoolQ train 1,945, SQuAD v2 train 500 binary and 200 Locate, Qasper train 20,
VQAv2 20 images) is for quantization calibration only (the weight conversion), never for evaluation.
`check_sets.py` fails if any calibration line shares a passage (also as a substring or by its first 120 characters),
a question, a dataset row, an image or a paper with an evaluation set; pass your own calibration files as arguments
to check them too. BoolQ train and validation share passages (15% of train), so a split alone is not
enough.

## 3. BF16 references of the original model

```sh
uv run python bench/quality/reference.py --model pplx-decider-flash-v1.3 --out bench/quality/out/flash-bf16.jsonl
uv run python bench/quality/reference.py --model pplx-decider-v1.3 --out bench/quality/out/27b-bf16.jsonl
```

The reference is [bf16ref](bf16ref/): the original checkpoint
([perplexity-ai/pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3) @ `d06fb3f2`,
[perplexity-ai/pplx-decider-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-v1.3) @ `9c2ebb27`, Apache-2.0),
unquantized, in plain MLX: BF16 matmuls, the Gated DeltaNet recurrence as a chunked FP32 reference, the Locate LoRA
added unmerged, one sequence at a time. Tokenization and image preprocessing are decider_mlx's, which reproduce
transformers bit for bit, so both sides see the same inputs. It was validated against the official PyTorch
implementation: 27B 100/100 gate decisions (mean |dp| 0.0014, Delta-logit sd 0.066, all feature probes identical),
Flash 99/100 (the one difference at an official probability of exactly 0.5). Its own error is an order of magnitude
below that of any quantized build.

Without `--original DIR` the checkpoint is downloaded into the Hugging Face cache (Flash 9.6 GB, 27B 54.1 GB).

**Memory.** Flash peaks at about 12 GB. **27B keeps the whole BF16 model resident and peaks at 57.1 GB (MLX)**,
which in practice exceeds what a 64 GB Mac can give one process; use a 96 GB or larger Mac, pass `--memory-gb 62`
(the default for 27B), and run nothing else meanwhile. `--memory-gb` is MLX's memory limit: an input whose estimated
peak would exceed it fails with a `MemoryError` row instead of swapping. Times on an M5 Ultra: Flash 182 s for the
core suite and about 160 s for the extra sets; 27B 635 s and about 550 s.

## 4. Run a build

```sh
uv run python bench/quality/run.py --model pplx-decider-flash-v1.3 --out bench/quality/out/flash.jsonl
uv run python bench/quality/run.py --model pplx-decider-v1.3 --out bench/quality/out/27b.jsonl
```

This runs the decider_mlx of the working tree, with the weights from the Hugging Face cache (fetched when missing),
or from `--weights DIR`, a local copy of a converted weights repository (for example one not uploaded yet). It
waits for the startup preparation, then sends each request alone as `app.decide(request, execution_mode=..., temperature=1.0)`.

Both runners write one row per request, `{"id", "set", "engine", "response" | "error", "elapsed_s",
"request_sha256"}`, and resume an interrupted run (ids already in `--out` are skipped). Common options:
`--sets core|extra|all|name,name`, explicit JSONL paths instead, `--limit N` (first N lines of each file) and
`--ids FILE` for smoke tests. A failing request becomes an `error` row; it is never dropped.

## 5. Compare

```sh
uv run python bench/quality/compare.py --ref bench/quality/out/flash-bf16.jsonl --cand bench/quality/out/flash.jsonl \
    --out bench/quality/out/flash-report.json
```

prints one line per set and question type and writes the full report (add `--rows` for every question). Rows are
matched by id; a candidate row whose `request_sha256` differs from the evaluation set's request is reported as an
error, not compared. `--subset` compares only the ids the candidate answered (partial runs).

| Type | Metrics |
|---|---|
| noul | agreement at 0.5 with Wilson 95% CI, mean/max \|dp\|, binary KL(ref \|\| cand), Delta-logit mean/sd/max, flips near the threshold (reference \|p - 0.5\| < 0.1) and elsewhere, flips expected from Gaussian logit noise of the measured sd, gold-label accuracy of both |
| choice | argmax agreement, max \|dp\| over the options, KL, near-tie flips (reference top-2 margin < 0.2), gold accuracy |
| score | argmax agreement, \|Delta score\| mean/max, KL, gold accuracy (BoolQ: score above the midpoint = yes) |
| locate | exact agreement of the span set, empty/non-empty agreement, top span text, character Jaccard, score difference of shared spans, spans only on one side and flips of spans scored below 0.7, gold EM/F1 of the top span (SQuAD v2, empty for unanswerable) |

**Two builds.** `--cand A.jsonl --cand-b B.jsonl` adds a paired bootstrap (2,000 resamples, 95% CI) of A - B for
mean |dp|, KL and agreement on the same questions. Use this, not the agreement counts, to decide between builds:
on 100 questions the agreement count moves by several flips from noise alone, while the paired mean |dp| CI is
about +-0.003 on the full suite. A difference whose CI includes zero means "equivalent".

**Locate as numbers.** Span agreement only sees spans crossing 0.5. Run both sides with `--locate-threshold 0.2`
(decisions are unaffected) and compare the span scores:

```sh
uv run python bench/quality/reference.py --model pplx-decider-flash-v1.3 --sets squad-locate100,squad-locate200b \
    --locate-threshold 0.2 --out bench/quality/out/flash-bf16-soft.jsonl
uv run python bench/quality/run.py --model pplx-decider-flash-v1.3 --sets squad-locate100,squad-locate200b \
    --locate-threshold 0.2 --out bench/quality/out/flash-soft.jsonl
uv run python bench/quality/locate_soft.py --ref bench/quality/out/flash-bf16-soft.jsonl \
    --cand bench/quality/out/flash-soft.jsonl --hard bench/quality/out/flash.jsonl
```

`--hard` checks that cutting the soft output at 0.5 gives the normal output back. `compare.py --locate-min-score 0.5`
reads soft outputs like normal ones.

**Resolution.** `resolution.py REF.jsonl` shows how many binary flips pure logit noise of a given sd would produce
on that reference (Poisson binomial), for example why 97/100 and 100/100 on the gate are not distinguishable at the
noise level of a 4-bit build.

## Reading the results

- The gate (boolq100, 96/100 or better in every execution condition) is a pass/fail regression check. It is also
  covered byte for byte by `tests/` (`DECIDER_MLX_TEST_MODELS`).
- Compare a build with the BF16 reference of the same model; Flash and 27B differ from each other far more than any
  build differs from its original (boolq100: 95/100).
- squad-choice60 is almost always answered with probability 1 and carries little information; score (74 questions)
  and choice (224) are small sets.
- The released builds measured with these tools: see [docs/models.md](../../docs/models.md).

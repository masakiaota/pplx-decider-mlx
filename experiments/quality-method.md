# Quality method

The question asked of every candidate was: how close are its outputs to the original v1.3 model? Accuracy against dataset labels is reported only as a secondary check, because a quantized engine can score higher on labels while drifting from the original (this happened for 27B Locate, see [27b.md](27b.md)).

## BF16 references

References are produced by an MLX implementation of the original model running the unquantized BF16 weights, with the reference (non-optimized) DeltaNet path.

- It was validated against the official PyTorch implementation. 27B: 100/100 decisions on the gate, mean |dp| 0.0014, max 0.031, Delta-logit standard deviation 0.066, and all feature probes (choice, score, Locate, images, isolated mode) identical. Flash: 99/100; the single difference is a question where the official probability is exactly 0.5.
- The 27B reference needs the whole BF16 model resident (MLX peak 57.1 GB) and therefore an exception to the 64 GB job cap on a larger machine. It produces all 772 evaluation requests in 10.6 minutes, about 25x faster than the official PyTorch implementation with offloading, so new references are cheap to generate.

The gate set is compared against stored official PyTorch outputs; everything else against the MLX BF16 references.

## Evaluation sets

| Set | Size | Content |
|---|---:|---|
| boolq100 | 100 | The gate; BoolQ validation, fixed IDs from earlier rounds |
| boolq-val300 | 300 | BoolQ validation excluding the gate, binary |
| boolq-choice-score60 | 60 | BoolQ validation, 3-way choice plus 5-level score |
| squad-choice60 | 60 | SQuAD v2, 4-way choice |
| squad-locate100 | 100 | SQuAD v2, Locate, 70 answerable and 30 unanswerable |
| squad-multi30 | 30 x 2 | Four binary questions per context, isolated and separate |
| vqa30 | 30 | VQAv2, choice plus binary, distinct images |
| qasper-long30 | 30 | Qasper yes/no on paper text cut to about 3K-28K characters (up to 5.9K tokens) |
| features16 | 16 x 2 | Feature probes with official references, both modes |
| squad-locate200b | 200 | Additional Locate set, disjoint from the above |
| qasper-long40b | 40 | Additional long set, 1.7K-7K tokens, papers disjoint from long30 |
| vqa60b | 60 | Additional image set, images disjoint |
| long 12K-32K | 9-10 | Concatenated Qasper papers, for long-input checks |

The first eleven files are 772 requests and 1,078 questions; with the additional sets 1,438 questions. Per-engine summaries usually cite the decision subsets (Flash: 826 binary, 224 choice, 74 score, 314 Locate; 27B harness: 964 questions) and the gate.

Each output row records the SHA-256 of the request it answered. This was added after an image set was rebuilt with the same IDs but different options; rows whose request hash does not match the evaluation set are reported as errors, not compared.

## Harness

For each candidate and reference pair the harness reports:

- decision agreement per question type (threshold 0.5 for binary, argmax for choice and score);
- mean and max |dp|, KL divergence, Delta-logit standard deviation;
- flips within |p - 0.5| < 0.1 of the threshold;
- for Locate, exact agreement of the span set (spans with score >= 0.5) and positions, and gold-label EM/F1;
- paired bootstrap confidence intervals (2,000 resamples) for differences between two candidates on the same questions.

**Continuous Locate comparison.** Span agreement is coarse: most disagreements are spans whose score sits at 0.51-0.57. Candidates and references are also run with the Locate threshold lowered to 0.2, so span scores can be compared as numbers. Cutting those outputs at 0.5 reproduces the normal outputs exactly. This exposed a systematic downward bias of -0.125 in one 27B candidate that span counts alone had shown only as "84/100".

## Gate

boolq100 at temperature 1, threshold 0.5, against official outputs, under four conditions: separate and isolated mode, each one request at a time and as one batch. The pass mark is 96/100 in every condition. It is a fast regression test; every shipped engine scores 100/100 in all four.

## Why boolq100 alone lacks resolution

The count of agreeing decisions on 100 questions is decided by the few questions whose probability is close to 0.5. Modelling a candidate's error as Gaussian noise on the logit with the candidate's measured standard deviation gives the expected number of flips by noise alone:

- At Delta-logit sd 0.86 (an early 27B W4A8 candidate), noise alone fails the gate (5 or more flips) with probability about 12%, and 12-18% across the W4A8 candidates.
- Results between 97/100 and 100/100 fall inside the 90% noise range at the same sd. Counts separate two candidates only when their error differs by about 2x.
- Two W4A8 weight sets whose mean |dp| differed by about 0.004 could not be told apart on 100 questions.

Paired continuous metrics on the larger sets fix this. The 95% CI half-width of a paired mean |dp| difference is about 0.0095 on 100 questions and about 0.003 on 1,124. On the larger sets the same two W4A8 weight sets differed significantly (+0.0039 [+0.0009, +0.0069]). Measured Delta-logit sd on all binary questions: Flash 8-bit 0.25, 27B B015 0.47, first and second 27B W4A8 weights 1.06 and 0.88. The error did not grow noticeably with input length (B015 0.29-0.53, W4A8 0.44-0.91, based on few long questions).

Decisions in this record therefore use: gate as a pass/fail check, decision agreement on the large sets, and the paired CI of mean |dp| and KL. A change was considered equivalent when the CI included zero.

## Calibration and evaluation de-duplication

Quantization calibration (27B rounding compensation, Flash score-layer selection) uses a separate pool: BoolQ train (1,945), SQuAD v2 train (500 binary, 200 Locate), Qasper train (20 long), VQAv2 (20 images). Every pool file is checked against every evaluation set by normalized passage hash, question hash, dataset index, image ID and hash, and paper ID; the check runs as a script and fails on any overlap.

Two findings shaped this:

- **Train and validation are not disjoint in BoolQ.** 1,439 of 9,427 BoolQ train rows (15%) share their passage with a validation row, some of them with gate questions. Splitting by dataset split is not enough.
- **Near duplicates.** The same Wikipedia paragraph also appears in longer and shorter versions across splits. Exact-match removal missed these; a prefix and substring check removed 55 more. The same check found 4 near duplicates in one experiment's calibration pool, which were reported to that experiment.

The earlier 27B compensation had used 1,062 tokens from the first four gate questions as calibration. That is both too little data and an overlap with the evaluation; the recalibration on about 240K non-overlapping tokens is described in [27b.md](27b.md).

## Limits

- Choice on squad-choice60 is nearly always at probability 1 and carries little information.
- Score (74 questions) and choice (224) sets are small.
- Raw logits are not compared directly; binary Delta-logit is used instead.
- Large real-photo images (0.6-1.6 MP) were checked only in a dedicated 120-row set.

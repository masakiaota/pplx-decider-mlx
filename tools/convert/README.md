# Building the weights from the original checkpoints

These scripts build the two Hugging Face repositories that decider-mlx downloads,
[masakiaota/pplx-decider-flash-v1.3-mlx](https://huggingface.co/masakiaota/pplx-decider-flash-v1.3-mlx) and
[masakiaota/pplx-decider-v1.3-mlx](https://huggingface.co/masakiaota/pplx-decider-v1.3-mlx), from the original
BF16 checkpoints [perplexity-ai/pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3)
@ `d06fb3f2` and [perplexity-ai/pplx-decider-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-v1.3) @
`9c2ebb27` (Apache-2.0). They are the scripts that made the published files, cleaned up, with the same arithmetic.
`verify.py` compares a result with the repository's `manifest.json`. Why each format was chosen is in
[experiments/](../../experiments/README.md) ([flash.md](../../experiments/flash.md), [27b.md](../../experiments/27b.md)).

Everything runs in the project environment (`uv sync`) from the repository root, on an Apple silicon Mac, with
`MLX_ENABLE_TF32=0` (set by the scripts). `calib_set.py` also needs `pyarrow` (`uv run --with pyarrow`).
Converted outputs and intermediate folders are large; put `WORK` on a disk with enough space.

```sh
uv run python tools/convert/originals.py flash --download      # or --original DIR for a local copy; checks sha256
uv run python tools/convert/originals.py 27b --download        # 54 GB
```

## Flash (about 1 minute, MLX peak 7 GB, 25 GB of disk)

```sh
O=$(uv run python -c "import sys; sys.path.insert(0, 'tools/convert'); from originals import locate; print(locate('flash'))")
W=WORK/flash
uv run python tools/convert/quantize.py "$O" $W/q8 --bits 8          # 1. affine 8-bit, group 64
uv run python tools/convert/flash_shared.py $W/q8 $W/shared          # 2. shared affine8 + row sums
uv run python tools/convert/flash_locate.py "$O" $W/locate           # 3. Locate LoRA merged, 8-bit
uv run python tools/convert/assemble.py flash "$O" $W/out --main $W/shared --locate $W/locate
uv run python tools/convert/verify.py $W/out path/to/pplx-decider-flash-v1.3-mlx/manifest.json
```

| Step | What it does |
|---|---|
| 1 `quantize.py --bits 8` | `mx.quantize(w, group_size=64, bits=8)` for every text linear layer except the DeltaNet gates `in_proj_a/b`; the rest stays BF16 |
| 2 `flash_shared.py` | MLP and input projections of layers 0-30 re-quantized per row to symmetric INT8 from the dequantized 8-bit values, stored as unsigned codes q+128 with the BF16 row scale repeated per group and bias -128 x scale, so MLX's 8-bit matmul and the INT8 x INT8 kernel read the same bytes; `row-sums.safetensors` |
| 3 `flash_locate.py` | `BF16(W + B A alpha/r)` from the original weights and the FP32 Locate LoRA, then affine 8-bit (gates stay BF16); marker embedding, head and config copied |
| `assemble.py` | the repository layout, unchanged original files, `mlx_config.json`, NOTICE, model card, `manifest.json` |

## 27B

The 368 row-wise 4-bit tensors (all text linear layers except k/v) are rounded with GPTQ-style error
compensation, which needs H = E[x x^T] of each layer's input over calibration data. The published H was collected
with a 4-bit **calibration model** (not the BF16 original), so reproducing the published bytes needs that model
too. The chain:

```
original BF16 ──quantize --bits 4──> Q4 ──calib_model (stage1.jsonl)──> calibration model
                                      │                                      │ gram (mix.jsonl)
                                      │                                      v
                                      ├──rest──> model-rest.safetensors      H ──rowq4 (+ original BF16)──> rowq4 ──s4──> rowq4/ (s4)
original locate-span/adapter ──adapter27b──> locate/adapter.safetensors
```

```sh
O=$(uv run python -c "import sys; sys.path.insert(0, 'tools/convert'); from originals import locate; print(locate('27b'))")
W=WORK/27b
uv run python bench/quality/download.py && uv run --with pyarrow python bench/quality/build_sets.py
uv run --with pyarrow python tools/convert/calib_set.py $W/calib                          # 4a
uv run python tools/convert/quantize.py "$O" $W/q4 --bits 4                                 # 1
uv run python tools/convert/calib_model.py "$O" $W/q4 $W/calib/stage1.jsonl $W/calib-model  # 4b (see MLX version below)
uv run python tools/convert/gram.py $W/calib-model $W/calib/mix.jsonl $W/gram               # 5
for i in 0 1; do OMP_NUM_THREADS=14 VECLIB_MAXIMUM_THREADS=14 \
  uv run python tools/convert/rowq4.py "$O" $(ls -d $W/gram/p* | paste -sd, -) $W/rowq4 --shard $i/2 & done; wait   # 6
uv run python tools/convert/s4.py $W/rowq4 $W/s4 --source-label experiments/night-20261119/D-002-full-convert/rowq4  # 7
uv run python tools/convert/rest.py $W/q4 $W/rest                                           # 8
uv run python tools/convert/adapter27b.py "$O" $W/adapter.safetensors                       # 9
uv run python tools/convert/assemble.py 27b "$O" $W/out --q4 $W/q4 --rest $W/rest --rowq4 $W/s4 --adapter $W/adapter.safetensors
uv run python tools/convert/verify.py $W/out path/to/pplx-decider-v1.3-mlx/manifest.json
```

| Step | What it does | Time (M5 Ultra) | Memory | Disk |
|---|---|---|---|---|
| 1 `quantize.py --bits 4` | affine 4-bit group 64, as for Flash | 1 min (reads 54 GB) | MLX 7 GB | 17 GB |
| 4a `calib_set.py` | 797 calibration requests (BoolQ, SQuAD v2, Qasper train; SQuAD Locate), about 240K tokens, checked against the evaluation sets; `stage1.jsonl` | seconds | small | small |
| 4b `calib_model.py` | 64-channel block GPTQ of gate/up in layers 0-62 of Q4, H from 4 requests | 5 min | 25 GB | 6 GB (APFS clone of Q4) |
| 5 `gram.py` | H of the 6 input groups of all 64 layers, 8 passes of 8 layers | 8 x 5.5 min | 36 GB | 100 GB |
| 6 `rowq4.py` | per-row scale search (r in 0.70-1.00), GPTQ with act-order and damping 0.03, float64 | 37 min (2 processes) | 15 GB RSS each | 12 GB |
| 7 `s4.py` | codes XOR 0x88888888 (offset-8 nibbles to two's complement) | 1 min | small | 12 GB |
| 8 `rest.py` | Q4 minus the 368 replaced tensors: k/v 4-bit and the BF16 tensors in one file | 30 s | small | 3.6 GB |
| 9 `adapter27b.py` | the Locate LoRA cast to BF16, marker kept FP32 | seconds | 2 GB | 0.9 GB |

The BF16 original is never loaded as a whole: steps 1, 4b and 6 read it one shard or one tensor at a time. A
64 GB Mac is enough for every step (run the steps one at a time).

**The calibration engine.** Steps 4b and 5 run the requests through `calib_engine/`, the research engine the
published weights were calibrated with, kept as it was (only its tokenizer is decider_mlx's, which gives the same
token ids as transformers, and image input is removed). It is not the decider-mlx engine: its activations must be
those that produced the published H, bit for bit. It is used for nothing else.

**MLX version of step 4b.** The calibration model was made with MLX 0.32.4; everything else with MLX 0.32.5 (the
version this project pins). On 0.32.5, step 4b gives factors that differ from layer 3, the first full-attention
layer, on (relative differences up to about 0.2%), so the calibration model, H and rowq4 differ slightly from the
published files. For a byte-identical rebuild, run step 4b with MLX 0.32.4:
`uv run --with mlx==0.32.4 python tools/convert/calib_model.py ...` (not tested). With either version the
resulting weights are a valid build of the same method.

**Metadata that is not arithmetic.** The published s4 files record `source:
experiments/night-20261119/D-002-full-convert/rowq4/<file>` in their metadata (the research path of their input),
hence `--source-label`; and the `safetensors` package that wrote them put the two metadata keys in an arbitrary
order per file, so `verify.py` accepts a header that differs only in that order when the parsed header and all
tensor bytes are identical. The index of `model-rest.safetensors` carries the note "B013 minus 368 row-Q4-replaced
tensors" (B013 is the research name of the calibration model). `mlx_config.json` records the MLX version of the
conversion (`verify.py` reports a difference in that field separately).

## What was verified

On a Mac Studio (M5 Ultra), MLX 0.32.5 built from source, numpy 2.5.5 (Accelerate), against the manifests of the
two repositories as uploaded:

| | Result |
|---|---|
| Flash, every step from the original | all 22 manifest files identical, except `mlx_version` in the two `mlx_config.json` (0.32.5 vs 0.32.4) |
| 27B `quantize` + `rest`, `adapter27b`, configuration files | identical (`model-rest.safetensors`, its index, the adapter, all original files; `mlx_config.json` up to `mlx_version`) |
| 27B `s4` from the research rowq4 files | all 368 identical in tensor bytes and parsed header; 220 byte for byte, 148 differ only in metadata key order |
| 27B `rowq4` from the archived research H, layers 0, 3 and 63 (16 tensors) | identical to the research rowq4 files, and after `s4` to the published files |
| 27B `calib_set` | `mix.jsonl`, `pool.jsonl`, `heldout.jsonl` byte-identical to the research files; `stage1.jsonl` the same 4 requests |
| 27B `calib_model` compensation, from the research factors | all 11 calibration-model shards identical to the research calibration model |
| 27B `calib_model` factors on MLX 0.32.5 | identical for layers 0-2, different from layer 3 on (see "MLX version of step 4b") |
| 27B `gram`, pass 0 (layers 0-7, all 797 requests) on the research calibration model | all 32 H files identical to the research H; `rowq4` of layer 0 from them identical to the published files |
| 27B whole chain on MLX 0.32.5, pass 0 | H of layers 0-2 identical, layer 3 on not (it follows the factors); `rowq4` of layer 0 identical |

Not run in full: `gram.py` passes 1-7 and `rowq4.py` for all 368 tensors (about 80 minutes together); step 4b with
MLX 0.32.4.

---
license: apache-2.0
base_model: perplexity-ai/pplx-decider-flash-v1.3
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-classification
tags:
  - mlx
  - apple-silicon
  - decider
  - classification
  - multimodal
  - 8-bit
---

# pplx-decider-flash-v1.3-mlx

Perplexity's [Decider Flash v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3), converted for Apple
silicon Macs (M5 or later) and run by [decider-mlx](https://github.com/masakiaota/pplx-decider-mlx), a local HTTP
server compatible with the official [Decisions API](https://docs.perplexity.ai/api-reference/decisions-post).

- **About 2x the official Mac app**: one request in 44.8 ms against 88.7 ms, and up to about 3.9x with decider-mlx's
  array batching (20 requests in 0.31 s).
- **Close to the original model**: the same answer as the original BF16 model on 821/826 binary, 222/224 choice and
  73/74 score questions held out from training.
- **Local extensions**: passage extraction (Locate), `isolated` mode and array requests, which the official API does
  not have.

> [!NOTE]
> This is an unofficial derivative. It is not affiliated with, endorsed by or supported by Perplexity AI; problems
> with this conversion belong in the decider-mlx issue tracker. The weights use a custom 8-bit layout and an INT8
> matrix product, so they run only with decider-mlx (a generic MLX loader will not reproduce the model).

## Comparison with the official Mac app

Processing time under the same HTTP workload:

| Case | Official Mac app | decider-mlx | Speedup |
|---|---:|---:|---:|
| 1 request (427 tokens) | 88.7 ms | 44.8 ms | 2.0x |
| 20 requests, one at a time | 1.21 s | 0.58 s | 2.1x |
| 20 requests, 4 concurrent | 1.19 s | 0.55 s | 2.2x |
| Long input (5.4K tokens) | 0.96 s | 0.42 s | 2.3x |
| 20 requests in one array | not supported | 0.31 s * | 3.9x (vs one at a time) |

- Measured on a Mac Studio (M5 Ultra), sending the same requests to freshly started servers.
- The official Mac app serves Flash v1.2 (`q8_0` GGUF); decider-mlx runs v1.3. The value marked * was measured
  separately on the release package.
- Startup is light too: about 0.5 s from process start to the first answer, against 2.2 s for the official app.

## Quick start

Requirements: a Mac with an M5 chip or later, 24 GB of memory or more recommended, macOS 27 or later, Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/masakiaota/pplx-decider-mlx && cd pplx-decider-mlx
uv sync
uv run decider-mlx fetch --model pplx-decider-flash-v1.3      # this repository, at a pinned revision
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
```

```sh
curl -s localhost:8000/v1/decisions -H 'Content-Type: application/json' -d '{
  "model": "pplx-decider-flash-v1.3",
  "state": "My integration keeps failing with a 401 since we rotated our API key.",
  "questions": {"urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"}}}'
```

The server takes the official Decisions API body (`POST /v1/decisions`). From Python:

```python
from decider_mlx import load
app = load("pplx-decider-flash-v1.3")
app.decide({"state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}})
```

`serve`, `run` and `load()` download this repository automatically when it is not in the Hugging Face cache, and
`decider-mlx fetch` checks every file against `manifest.json` (sha256). See the decider-mlx
[usage guide](https://github.com/masakiaota/pplx-decider-mlx/blob/main/docs/usage.md) for batching, isolated mode,
temperature, Locate and image inputs.

## Speed and memory (Mac Studio, M5 Ultra, decider-mlx 0.1)

| Case | Time |
|---|---:|
| Process start to accepting HTTP requests | 0.44-0.47 s |
| Process start to the first answer (427 tokens) | 0.51-0.54 s |
| Single request, 427 tokens (repeated) | 43 ms |
| 20 requests in one array | 0.31 s |
| 520 mixed requests, one at a time over HTTP | 26.5 s (median 32 ms) |
| Long input: 5.4K / 12.9K / 32.4K tokens | 0.42 / 2.17 / 7.03 s |
| 1,072 mixed requests in arrays (bulk packing) | 47 s |

Startup preparation (compiling the graphs for common shapes) runs in the background between requests and is done
after about 5.4 s; answers do not depend on it.

Memory: 5.74 GB of memory-mapped main weights, plus 3.80 GB after the first Locate request. MLX peak 14.2 GB on a
32.4K-token Locate request after Locate is loaded (11.2 GB at 32.4K without Locate); about 7.2 GB RSS.

## Agreement with the original model

Measured against BF16 reference outputs of the original model (an MLX port that matches the official PyTorch
implementation), on held-out evaluation sets of about 1,000 questions:

| Question type | Same answer as BF16 |
|---|---|
| Binary (`noul`) | 821/826 |
| `choice` | 222/224 |
| `score` (argmax) | 73/74 |
| Locate span sets | 305/314 |
| Long inputs, 12K-32K tokens | 9/9 |
| BoolQ gate, 4 execution conditions | 100/100 each |

Probabilities differ slightly from BF16 (quantization noise). Method and per-technique results:
[experiments/](https://github.com/masakiaota/pplx-decider-mlx/tree/main/experiments).

## Conversion

| Part | Format |
|---|---|
| Language-model linear layers | 8-bit, one weight file used two ways ("shared affine8"): unsigned codes with one scale per output row, described to MLX as group-64 affine 8-bit. The same bytes feed MLX's quantized matmul and decider-mlx's INT8 x INT8 matmul (`row-sums.safetensors` holds the per-row code sums for its epilogue). |
| Embedding, normalization, convolution, vision tower, readout head | unchanged from the original (BF16) |
| Locate (`locate/`) | the original rank-64 Locate LoRA merged into the linear weights in FP32, rounded to BF16, then quantized to affine 8-bit (group 64); span head and marker embedding unchanged. Loaded only on the first Locate request. |
| Tokenizer, processor, `config.json`, `decision_config.json`, `joint_config.json`, chat template | unchanged copies of the original repository |

At run time, decider-mlx quantizes activations per row to INT8 for the MLP and input projections. Sequences with a
`score` question use MLX's standard 8-bit matmul in layers 0-15, because INT8 error on score distributions comes from
early layers. Locate runs on the standard 8-bit path.

Main weights: 5.74 GB. Locate weights: 3.80 GB. The original repository is about 8.5 GiB of BF16 weights.

The scripts that build these files from the original checkpoint are in [tools/convert](https://github.com/masakiaota/pplx-decider-mlx/tree/main/tools/convert) of decider-mlx.

## Files

`manifest.json` lists the size and sha256 of every file except this README. `mlx_config.json` describes the
quantization. `LICENSE` is the original Apache-2.0 license, and `NOTICE` keeps the original notice and lists the
modifications.

## License

Apache-2.0, as the original model. The original model is derived from Qwen/Qwen3.5-4B (Apache-2.0); see `NOTICE`.

# decider-mlx

English | [日本語](README.ja.md)

**decider-mlx** is an [MLX](https://github.com/ml-explore/mlx)-based inference engine that runs Perplexity's
classification models **Decider v1.3 (Flash / 27B)** locally on Apple silicon Macs (M5 or later). It provides an HTTP
server compatible with the official [Decisions API](https://docs.perplexity.ai/api-reference/decisions-post), plus
JSONL batch processing and a Python API, so existing code can use it by pointing at your Mac.

- **About 2x the official Mac app**: Flash answers about twice as fast as the official Mac app. With this engine's
  array batching, it is up to about 3.9x faster.
- **27B and local extensions**: the **27B model**, which the official Mac app does not offer, runs in less than half
  the memory of the original BF16 model. Passage extraction (Locate) and `isolated` mode, which the official API does
  not have, are also available.
- **Close to the original model**: custom INT8 and W4A8 Metal kernels make it fast, and held-out evaluation shows
  high agreement with the original BF16 models.

> [!NOTE]
> This is an unofficial project. It is not affiliated with, endorsed by or supported by Perplexity AI. It runs
> converted weights (8-bit for Flash, row-wise 4-bit for 27B), so the official BF16 weights cannot be used as they
> are, and its probabilities differ slightly from the original model's (see [Quality](#quality)).

## Comparison with the official Mac app

Processing time of Flash under the same HTTP workload, against the official Mac app:

| Case (Flash, HTTP) | Official Mac app | decider-mlx | Speedup |
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

### Requirements

- **Chip**: Apple silicon **M5 or later** (M4 or earlier may work but is not supported)
- **OS**: macOS 27 or later
- **Recommended memory**: 24 GB or more (Flash) / 36 GB or more (27B)
- **Free disk space**: 9.6 GB or more (Flash) / 16.7 GB or more (27B)
- **Tool**: [uv](https://docs.astral.sh/uv/)

### Install and start the server

```sh
git clone https://github.com/masakiaota/pplx-decider-mlx
cd pplx-decider-mlx
uv sync
uv run decider-mlx fetch --model pplx-decider-flash-v1.3   # downloads the weights (9.6 GB)
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
```

> `uv sync` downloads MLX 0.32.5, which is not on PyPI yet, from this repository's
> [GitHub Release](https://github.com/masakiaota/pplx-decider-mlx/releases/tag/mlx-0.32.5-fc344c9), so the first run
> may take a little longer (details in [CONTRIBUTING.md](CONTRIBUTING.md)).

### Send the first request

From another terminal:

```sh
curl -s localhost:8000/v1/decisions -H 'Content-Type: application/json' -d '{
  "model": "pplx-decider-flash-v1.3",
  "state": "My integration keeps failing with a 401 since we rotated our API key. Please help.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "Charges and refunds", "support": "Technical integration errors",
                          "sales": "Questions about buying a product"}},
    "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?"},
    "frustration": {"type": "score", "criteria": ["calm", "mildly annoyed", "frustrated", "angry"]}
  }}'
```

Response (numbers rounded):

```json
{"model": "pplx-decider-flash-v1.3",
 "answers": {"team": {"type": "choice", "probabilities": {"billing": 0.0004, "support": 0.9992, "sales": 0.0004},
                      "choice": "support", "confidence": 0.9989},
             "urgent": {"type": "noul", "noul": 0.3847},
             "frustration": {"type": "score", "probabilities": {"0": 0.1221, "1": 0.3125, "2": 0.5492, "3": 0.0162},
                             "legend": {"0": "calm", "1": "mildly annoyed", "2": "frustrated", "3": "angry"},
                             "score": 1.4594, "confidence": 0.4271}},
 "usage": {"input_tokens": 288, "output_tokens": 0, "sequences": 3}}
```

Decider answers questions about text or images (`state`) with probabilities and scores, for tasks such as routing
requests, judging urgency and classifying documents. There are three basic question types (plus `locate`, a local
extension):

- **`choice`**: the probability of each option and the most likely option.
- **`noul`**: the probability of "yes" (true).
- **`score`**: the probability of each level and the expected level.

## Models and performance

Two models are available, depending on the task and the memory of your Mac. Choose **Flash** for quick answers and
high throughput, and **27B** for tasks that need the capacity of a 27B model.

| | Flash | 27B |
|---|---|---|
| `--model` | `pplx-decider-flash-v1.3` | `pplx-decider-v1.3` |
| Original model | [perplexity-ai/pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3) | [perplexity-ai/pplx-decider-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-v1.3) |
| Weights for this engine | [8-bit / 9.6 GB](https://huggingface.co/masakiaota/pplx-decider-flash-v1.3-mlx) | [4-bit / 16.7 GB](https://huggingface.co/masakiaota/pplx-decider-v1.3-mlx) |
| Recommended memory | 24 GB or more | 36 GB or more |
| Process start to accepting requests | 0.4-0.5 s | 0.3-0.5 s |
| Process start to the first answer | 0.5 s | 0.6-1.0 s |
| 1 request (427 tokens) | 43 ms | 220 ms |
| 20 requests in one array | 0.31 s | 1.92 s |
| Long input (32.4K tokens) | 7.0 s | 22.9 s |
| 1,072 mixed requests in arrays | 47 s | 236 s |
| MLX peak memory (after the HTTP benchmark) | 14.2 GB | 23.7 GB |

Measured over HTTP on a Mac Studio (M5 Ultra). The original 27B BF16 model peaks at 57.1 GB of memory; with row-wise
4-bit weights this engine needs 23.7 GB after the HTTP benchmark (25.0 GB for a 32K-token Locate request). Details
per model are in [docs/models.md](docs/models.md).

### Quality

How often the answer is the same as the original model's (BF16) reference output, on evaluation data not used for
training:

| Same answer as BF16 | Flash | 27B |
|---|---:|---:|
| Binary (`noul`) | 821 / 826 | 717 / 726 |
| Choice (`choice`) | 222 / 224 | 161 / 164 |
| Score (`score`, most likely level) | 73 / 74 | 70 / 74 |
| Locate (sets of extracted spans) | 305 / 314 | 100 / 114 |
| BoolQ gate (4 execution conditions) | 100 / 100 each | 100 / 100 each |

To compare with the original model on your own machine, see [bench/quality/](bench/quality/README.md).

## Other ways to use it

Besides the HTTP server, decider-mlx processes JSONL files and can be called from Python.

### JSONL files

```sh
uv run decider-mlx run --model pplx-decider-flash-v1.3 --input requests.jsonl --output responses.jsonl
```

### Python API

```python
from decider_mlx import load

app = load("pplx-decider-flash-v1.3")
response = app.decide({
    "state": "The museum opened in 1931.",
    "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}},
})
```

More options and details are in the [usage guide](docs/usage.md).

## Documentation

- [Usage guide](docs/usage.md): the HTTP server in detail, images, local extensions such as arrays and Locate, the
  Python API, running two models, offline use
- [API differences](docs/api.md): every difference from the official Decisions API
- [Models and measurements](docs/models.md): design, memory and benchmarks of Flash and 27B
- [Quality evaluation](bench/quality/README.md): how to compare with the original BF16 models
- [Development and tests](CONTRIBUTING.md): running the tests, speed benchmarks, notes on the MLX 0.32.5 wheels
- [Research record](experiments/README.md) / [weight conversion](tools/convert/README.md): what was tried and
  measured, and how to build the weights from the original BF16 checkpoints

## License

- The code in this repository is Apache-2.0 ([LICENSE](LICENSE)).
- Parts of the Metal kernels are adapted from MLX and mxfp4-int8-m5, both MIT ([NOTICE](NOTICE)).
- The test data in `tests/golden` contains BoolQ (CC BY-SA 3.0) and SQuAD (CC BY-SA 4.0) excerpts
  ([tests/golden/README.md](tests/golden/README.md)).
- The converted model weights are Apache-2.0 derivatives of Perplexity's models; see the `NOTICE` in each Hugging
  Face repository.

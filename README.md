# decider-mlx

English | [日本語](README.ja.md)

A fast local inference engine for Perplexity's Decider v1.3 classification models on Apple silicon Macs, built on
[MLX](https://github.com/ml-explore/mlx). It serves the official Decisions API body over HTTP, runs JSONL files, and
has a Python API.

| Model (`--model`) | Original | Weights for this engine | Recommended memory |
|---|---|---|---|
| `pplx-decider-flash-v1.3` | [perplexity-ai/pplx-decider-flash-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.3) | [masakiaota/pplx-decider-flash-v1.3-mlx](https://huggingface.co/masakiaota/pplx-decider-flash-v1.3-mlx) (8-bit, 9.6 GB) | 24 GB or more |
| `pplx-decider-v1.3` (27B) | [perplexity-ai/pplx-decider-v1.3](https://huggingface.co/perplexity-ai/pplx-decider-v1.3) | [masakiaota/pplx-decider-v1.3-mlx](https://huggingface.co/masakiaota/pplx-decider-v1.3-mlx) (4-bit, 16.7 GB) | 36 GB or more |

**This is an unofficial project. It is not affiliated with, endorsed by or supported by Perplexity AI.** The engine
runs converted weights (8-bit for Flash, row-wise 4-bit for 27B) with custom INT8 and W4A8 Metal kernels. It does
not run the official BF16 weights, and its probabilities differ slightly from the original model's (see
[Quality](#quality)).

On a Mac Studio (M5 Ultra):

| | Flash | 27B |
|---|---:|---:|
| Process start to accepting requests | 0.4-0.5 s | 0.3-0.5 s |
| Process start to the first answer | 0.5 s | 0.6-1.0 s |
| One request, 427 tokens | 43 ms | 220 ms |
| 20 requests in one array | 0.31 s | 1.92 s |
| Long input, 32.4K tokens | 7.0 s | 22.9 s |
| 1,072 mixed requests in arrays | 47 s | 236 s |
| MLX peak memory (HTTP benchmark) | 14.2 GB | 23.7 GB |

Flash is about 2x the official Mac app (Flash v1.2, q8_0 GGUF) on the same HTTP workload.

## Requirements

- A Mac with an **M5 chip or later**. The kernels and the length tables were tuned and tested on M5; earlier chips
  are not supported.
- Memory: 24 GB or more for Flash, 36 GB or more for 27B. Running both at once needs the sum.
- macOS with **Xcode and the Metal Toolchain**: MLX 0.32.5 is built from source during installation (see below).
  `xcodebuild -showComponent MetalToolchain` should report `installed`; if not, run
  `xcodebuild -downloadComponent MetalToolchain`.
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.14 when needed).
- Disk: 9.6 GB (Flash) or 16.7 GB (27B) in the Hugging Face cache.

## Install

```sh
git clone https://github.com/masakiaota/pplx-decider-mlx
cd pplx-decider-mlx
uv sync
```

`uv sync` builds MLX 0.32.5 from source at ml-explore/mlx commit `fc344c9`, pinned in `pyproject.toml`. That
version includes [ml-explore/mlx#4682](https://github.com/ml-explore/mlx/pull/4682) (faster quantized matrix
products on M5) and is not on PyPI yet. The first build takes about 2 minutes and needs network access; later syncs
reuse uv's cache. When `mlx==0.32.5` is published on PyPI, this project will switch to it.

Commands below use `uv run decider-mlx`; `source .venv/bin/activate` lets you type `decider-mlx` directly.

## Download the weights

```sh
uv run decider-mlx fetch --model pplx-decider-flash-v1.3
uv run decider-mlx fetch --model pplx-decider-v1.3
```

`fetch` downloads the converted weights, tokenizer, processor and configs from the Hugging Face repository of the
model at a revision pinned in this package, into the standard Hugging Face cache (`~/.cache/huggingface`, or
`HF_HUB_CACHE` / `HF_HOME`). It shows progress and the remaining time, with usage tips while you wait, and then checks
every file against the repository's `manifest.json` (sha256). Run it again at any time to re-check the files.

`serve`, `run` and `load()` fetch the weights automatically when they are missing, so this step is optional. With
`HF_HUB_OFFLINE=1` nothing is downloaded and the cache is used as it is.

## Serve

```sh
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
```

The server listens on `127.0.0.1` at once and answers `503` (with `Retry-After: 1`) until the weights are mapped,
about half a second later. It then prepares the compiled graphs for common request shapes in the background, between
requests, and shows its state on one line, for example
`pplx-decider-flash-v1.3: preparing 41/101 (accepting requests)`. Requests are answered during the preparation, and
the answers do not depend on it. `GET /metrics` returns the state and the memory use.

The only options are the model and the port. There is no memory limit or preflight check; MLX's defaults apply.

## HTTP

The request body is the official [Decisions API](https://docs.perplexity.ai/api-reference/decisions-post) body,
and `model` must be the served model's name.

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

(Numbers rounded here.) Errors use the official shape, `{"error": {"message", "type", "code", "param"}}`, with status
400 for invalid requests, 404/405 for other paths and methods, 413 over 32 MiB and 500 for internal failures. A model
name other than the served one, including `-latest` aliases and `perplexity/` prefixes, is a 400.

Images are OpenAI-style parts anywhere in `state`; they are placed before the text, as in the official library:

```json
{"model": "pplx-decider-flash-v1.3",
 "state": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
           {"type": "text", "text": "Customer photo of the damaged parcel."}],
 "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel visibly damaged?"}}}
```

PNG, JPEG and WebP data URLs, up to 8 images per request. After the images are taken out, the rest of `state` is
read as sent: a string as text, an object or array as JSON.

## Local extensions

These go beyond the official API. [docs/api.md](docs/api.md) lists every difference.

- **Arrays.** POST a JSON array of bodies and get a same-order array back. An invalid item gets
  `{"error": ...}` in its place and the others are answered. Arrays of 64 or more requests are packed into fixed
  2,048-token batches internally, which is the fastest way to run bulk work.
- **`?mode=isolated`.** Reads the state once per request and answers every question from it (the training-time
  layout). Cheaper for many questions on a long state; answers can shift slightly with the set of questions. The
  default, `separate`, answers each question in its own sequence.
- **`?temperature=`.** Without it, the model's saved temperature is used (Flash 1.3303, 27B 1.2264), as in the
  official API. `?temperature=1` gives the uncalibrated probabilities.
- **`?raw=1`.** Raw readout logits per question instead of answers.
- **Locate.** `{"type": "locate", "instructions": "error codes"}` returns verbatim spans of the state with character
  offsets and scores. The Locate weights are loaded on the first Locate request (Flash: +3.8 GB).

```sh
curl -s 'localhost:8000/v1/decisions?mode=isolated&temperature=1' -H 'Content-Type: application/json' -d '[
  {"model": "pplx-decider-flash-v1.3", "state": "The museum opened in 1931.",
   "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}}},
  {"model": "pplx-decider-flash-v1.3", "state": "Error E401 after the key rotation.",
   "questions": {"code": {"type": "locate", "instructions": "error codes"}}}]'
```

## Files in, files out

```sh
uv run decider-mlx run --model pplx-decider-flash-v1.3 --input requests.jsonl --output responses.jsonl
```

Each input line is a request (`{"state", "questions"}`; `model` is not needed), or
`{"id": ..., "request": {...}, "options": {"execution_mode": "isolated", "temperature": 1.0, "raw": false}}` with
options for that line. Each output line is `{"id", "index", "response"}` or
`{"id", "index", "error": {"type", "message"}}`, in input order. Files of 64 or more lines are packed like a large
array. The exit status is 1 when any line failed.

## Python

```python
from decider_mlx import load

app = load("pplx-decider-flash-v1.3")      # returns once the weights are mapped
response = app.decide({"state": "The museum opened in 1931.",
                       "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}}})
responses = app.decide([request1, request2], execution_mode="isolated", temperature=1.0)
logits = app.decide(request1, return_logits=True)
results = app.decide_each([request1, bad_request])   # a response or an exception per request
app.wait_ready()                                   # only for measurements: waits for the background preparation
```

`decide` is thread-safe: calls from any thread are served one at a time in arrival order by the model's worker
thread. It raises `decider_mlx.RequestError` for an invalid request and `decider_mlx.LengthError` when a question with
the state does not fit in 32,768 tokens (nothing is truncated). Images go in `"images"` as file paths, data URLs or
PIL images. Import `decider_mlx` before `mlx`. The library writes nothing to the terminal; it logs with `logging`.

## Two models at once

Run one server per model:

```sh
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
uv run decider-mlx serve --model pplx-decider-v1.3 --port 8001
```

The two processes are independent (no lock, no shared files) and can run requests at the same time; the GPU is
shared. A cross-process lock was measured 11-35% slower than running both freely.

## Quality

Measured against BF16 reference outputs of the original models on held-out evaluation sets:

| Same answer as BF16 | Flash | 27B |
|---|---:|---:|
| Binary (`noul`) | 821/826 | 717/726 |
| `choice` | 222/224 | 161/164 |
| `score` (argmax) | 73/74 | 70/74 |
| Locate span sets | 305/314 | 100/114 |
| BoolQ gate, 4 execution conditions | 100/100 each | 100/100 each |

Per-model details: [docs/models.md](docs/models.md).

## Development

```sh
uv sync                                         # includes pytest
uv run pytest                                   # model-free tests (API, HTTP checks, fetch against a mock Hub)
DECIDER_MLX_TEST_MODELS=pplx-decider-flash-v1.3,pplx-decider-v1.3 HF_HUB_OFFLINE=1 uv run pytest
```

With `DECIDER_MLX_TEST_MODELS`, the tests also load the models from the Hugging Face cache and compare BoolQ in
four execution conditions and every HTTP feature case byte for byte with `tests/golden` (see
[tests/golden/README.md](tests/golden/README.md)). The golden outputs were recorded on an M5 Ultra; other chips or
MLX builds may differ in the last bits.

```sh
uv run python bench/speed.py --model pplx-decider-flash-v1.3
```

prints in-process timings (single requests, a 20-request array, long inputs). `tests/hub_mock.py` is a local
stand-in for the Hugging Face Hub that serves a folder as a model repository, for testing `fetch` offline.

## How it was built

[experiments/](experiments/README.md) is the research record: what was tried for each model, what was kept or
dropped and why, with the measurements (weights, INT8 and W4A8 kernels, length buckets, bulk packing, startup).

## License

Apache-2.0 ([LICENSE](LICENSE)). Parts of the Metal kernels are adapted from MLX and from mxfp4-int8-m5, both MIT
([NOTICE](NOTICE)). The test data in `tests/golden` contains BoolQ (CC BY-SA 3.0) and SQuAD (CC BY-SA 4.0) excerpts
([tests/golden/README.md](tests/golden/README.md)). The model weights are Apache-2.0 derivatives of Perplexity's
models; see each Hugging Face repository's NOTICE.

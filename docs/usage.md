# Usage guide

English | [日本語](usage.ja.md)

The HTTP server, the local extensions, JSONL processing, the Python API and the management of the weights in
detail. Every difference from the official Decisions API is listed in [api.md](api.md).

## HTTP server

```sh
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
```

- **Startup and background preparation**: the server listens on `127.0.0.1` at once and answers `503` (with
  `Retry-After: 1`) until the weights are mapped, about half a second later. It then prepares the compiled graphs for
  common request shapes in the background, between requests, and shows its state on one line, for example
  `pplx-decider-flash-v1.3: preparing 41/101 (accepting requests)`. Requests are answered during the preparation, and
  the answers do not depend on it.
- **State and options**: `GET /metrics` returns the state of the preparation and the memory use. The only options
  are `--model` and `--port`.
- **Official API body**: the request body is the official
  [Decisions API](https://docs.perplexity.ai/api-reference/decisions-post) body, and `model` must be the served
  model's name (`-latest` aliases, `perplexity/` prefixes and other model names are a `400`).
- **Errors**: in the official shape, `{"error": {"message", "type", "code", "param"}}`: `400` for invalid requests,
  `404` / `405` for other paths and methods, `413` for bodies over 32 MiB and `500` for internal failures.

### Requests with images

Images are OpenAI-style parts anywhere in `state` (PNG, JPEG and WebP base64 data URLs, up to 8 images per
request). As in the official library, they are placed before the text; after the images are taken out, the rest of
`state` is read as sent (a string as text, an object or array as JSON).

```json
{"model": "pplx-decider-flash-v1.3",
 "state": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
           {"type": "text", "text": "Customer photo of the damaged parcel."}],
 "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel visibly damaged?"}}}
```

## Local extensions

Features for local use that the official Decisions API does not have.

- **Arrays**: POST a JSON array of bodies and get a same-order array back. An invalid item gets `{"error": ...}` in
  its place and the others are answered. Arrays of 64 or more requests are packed into fixed 2,048-token batches
  internally, which is the fastest way to run bulk work.
- **`?mode=isolated`**: reads the state once per request and answers every question from it (the training-time
  layout). Faster for many questions on a long state; answers can shift slightly with the set of questions. The
  default, `separate`, answers each question in its own sequence.
- **`?temperature=`**: without it, the model's saved temperature is used (Flash `1.3303`, 27B `1.2264`), as in the
  official API. `?temperature=1` gives the uncalibrated probabilities.
- **`?raw=1`**: raw readout logits per question instead of answers.
- **Locate**: `{"type": "locate", "instructions": "error codes"}` returns verbatim spans of the state with character
  offsets (start and end) and scores. The Locate weights are loaded on the first Locate request (Flash: about
  3.8 GB, 27B: about 0.95 GB).

```sh
curl -s 'localhost:8000/v1/decisions?mode=isolated&temperature=1' -H 'Content-Type: application/json' -d '[
  {"model": "pplx-decider-flash-v1.3", "state": "The museum opened in 1931.",
   "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}}},
  {"model": "pplx-decider-flash-v1.3", "state": "Error E401 after the key rotation.",
   "questions": {"code": {"type": "locate", "instructions": "error codes"}}}]'
```

## JSONL files

```sh
uv run decider-mlx run --model pplx-decider-flash-v1.3 --input requests.jsonl --output responses.jsonl
```

- **Input**: each line is a request (`{"state", "questions"}`; `model` is not needed), or
  `{"id": ..., "request": {...}, "options": {"execution_mode": "isolated", "temperature": 1.0, "raw": false}}` with
  options for that line.
- **Output**: one line per input line, in input order: `{"id", "index", "response"}` or
  `{"id", "index", "error": {"type", "message"}}`.
- **Behaviour**: files of 64 or more lines are packed like a large array. The exit status is `1` when any line
  failed.

## Python API

```python
from decider_mlx import load

app = load("pplx-decider-flash-v1.3")      # returns once the weights are mapped
response = app.decide({
    "state": "The museum opened in 1931.",
    "questions": {"a": {"type": "noul", "instructions": "Did it open before 1950?"}},
})
responses = app.decide([request1, request2], execution_mode="isolated", temperature=1.0)
logits = app.decide(request1, return_logits=True)
results = app.decide_each([request1, bad_request])   # a response or an exception per request
app.wait_ready()                                     # only for measurements: waits for the background preparation
```

- `decide` is thread-safe: calls from any thread are served one at a time in arrival order by the model's worker
  thread.
- It raises `decider_mlx.RequestError` for an invalid request and `decider_mlx.LengthError` when a question with the
  state does not fit in 32,768 tokens (nothing is truncated).
- Images go in the request's `"images"` as file paths, data URLs or PIL images.
- **Import `decider_mlx` before `mlx`**, so that the settings it needs are in place. The library writes nothing to
  the terminal; it logs with the standard `logging`.

## Running two models at once

Start one server per model, on different ports:

```sh
uv run decider-mlx serve --model pplx-decider-flash-v1.3 --port 8000
uv run decider-mlx serve --model pplx-decider-v1.3 --port 8001
```

The two processes are independent (no lock, no shared files) and can run requests at the same time, sharing the GPU
(the memory of both models is needed).

## The CLI and the weights

```sh
uv run decider-mlx fetch --model pplx-decider-flash-v1.3
uv run decider-mlx fetch --model pplx-decider-v1.3
```

- **Shorter commands**: after `source .venv/bin/activate`, type `decider-mlx` without `uv run`.
- **Download and cache**: `fetch` downloads the converted weights, tokenizer, processor and configs from each model's
  Hugging Face repository at the revision pinned in this package, into the standard Hugging Face cache
  (`~/.cache/huggingface`, or `HF_HUB_CACHE` / `HF_HOME`). `serve`, `run` and `load()` download the weights
  automatically when they are missing, so running `fetch` first is optional.
- **Progress and verification**: while downloading, `fetch` shows progress, the remaining time and usage tips (in
  Japanese when the system language is Japanese, otherwise in English). Afterwards it checks every file against the
  repository's `manifest.json` (sha256). Run it again at any time to re-check the files.
- **Offline use**: with `HF_HUB_OFFLINE=1` nothing is downloaded and only the local cache is used.

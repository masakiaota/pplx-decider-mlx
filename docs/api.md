# API: differences from the official Decisions API and library

decider-mlx accepts the body of the official [Decisions API](https://docs.perplexity.ai/api-reference/decisions-post)
(`POST /v1/decisions`) and answers with the official response shape. This page lists everything that differs: what
is not available locally, what is stricter, and the local extensions. The Python API is compared with the official
`pplx_decider` library at the end.

## Server

| | Official API | decider-mlx |
|---|---|---|
| Address | `https://api.perplexity.ai/v1/decisions` | `http://127.0.0.1:<port>/v1/decisions`, local only, no authentication |
| Models | several, with aliases | one model per server process, chosen at start (`--model`) |
| `model` field | official names and aliases | must be the served model's exact name (`pplx-decider-flash-v1.3` or `pplx-decider-v1.3`); `-latest` aliases, `perplexity/` prefixes and the other model are 400 |
| Response `model` | the model that answered | the served model's name (the same string as the request) |
| Startup | — | the port is open at once; `503` with `Retry-After: 1` until the weights are mapped (about 0.5 s) |
| Concurrency | — | any number of connections; requests run one at a time in arrival order on the model's worker thread |
| `GET /metrics` | — | local extension: load and preparation state, MLX memory (`active_bytes`, `peak_bytes`, `cache_bytes`, `mapped_bytes`) and RSS |

## Request body

| Field | Official API | decider-mlx |
|---|---|---|
| `model` | required | required, see above |
| `state` | string, object or array | the same. Images are taken out first (below); the rest is read as sent: a string is text, an object or array is serialized as JSON. Text parts left in an array are not joined. |
| `questions` | `choice`, `noul`, `score` | the same, plus `locate` (below) |
| `mode` | `web_grounding` | not available locally: `web_grounding` and any other value are 400 |
| `max_staleness` | web grounding freshness | accepted and ignored (no web data) |
| other fields | — | 400 `Unknown field: <name>` |

Question rules checked at the HTTP layer, with the official library's wording where it has one:

- `choice`: `criteria` maps option names to descriptions (or `null`), 1 to 255 options.
- `noul`: needs `instructions` or `criteria` (`{"true": ..., "false": ...}`).
- `score`: `criteria` is a list of 2 to 10 levels, lowest first (the API allows at most 10; the model needs at least
  2). The Python API accepts up to 255 levels, as the official library.
- An unknown `type` is 400 with `param` `questions.<id>.type`.

## Images

Images are OpenAI-style parts anywhere in `state`, at any depth:

```json
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
```

- Only base64 data URLs of PNG, JPEG and WebP. Remote `http(s)` URLs are not fetched (400).
- At most 8 images per request.
- They are taken out of `state` in reading order and placed before the text of the request, as in the official
  library's `images` field. Each image is resized, keeping its aspect ratio, to at most 1,638,400 pixels (the
  model's saved `max_pixels`). A 1.6 MP image uses about 1,600 of the 32,768 positions.
- What is left of `state` is read as sent. `[image, {"type": "text", "text": "Is this a cat?"}]` leaves
  `[{"type": "text", "text": "Is this a cat?"}]`, which is serialized as JSON and read as that text, as in the
  official library. Text parts are not joined. A plain string `state` cannot carry images over HTTP; in Python,
  pass the images in `"images"` and the text as `"state"`.

## Response

The same as the official API: `{"answers": {<id>: {...}}, "usage": {"input_tokens", "output_tokens": 0,
"sequences"}, "model"}`.

| Type | Answer |
|---|---|
| `choice` | `choice`, `probabilities` (per option), `confidence` |
| `noul` | `noul` = P(true) |
| `score` | `score` (expected level index), `probabilities`, `legend`, `confidence` |
| `locate` | `spans`: `start`, `end` (character offsets, end exclusive), `text`, `score` |

Probabilities come from the converted weights and differ slightly from the official model's
([models.md](models.md)).

## Errors

`{"error": {"message", "type", "code": null, "param"}}`, as the official API.

| Status | When |
|---|---|
| 400 `invalid_request_error` | invalid body or question, wrong model, unknown field or query parameter, a question that does not fit in 32,768 tokens with the state and images (nothing is truncated) |
| 404 | any other path |
| 405 (empty body, `Allow: POST`) | another method on `/v1/decisions` |
| 413 | body over 32 MiB |
| 500 `api_error` | an internal failure |
| 503 `service_unavailable` | the model is still loading (`Retry-After: 1`) |

Messages carry no Python class names.

## Local extensions

### Arrays

A JSON array of request bodies returns a JSON array of the same length and order. Each element is a response or,
for an invalid item, `{"error": {...}}`; the other items are still answered and the status is 200. An empty array is
400. Arrays of 64 or more requests are packed internally into fixed 2,048-token batches. Packing does not change any
answer.

### Query parameters

| Parameter | Values | Default |
|---|---|---|
| `mode` | `separate`: each question in its own sequence with the state (the official default). `isolated`: the state is read once per request and every question attends to it and to itself (the training-time layout; cheaper for many questions on a long state, answers can shift slightly with the set of questions) | `separate` |
| `temperature` | a positive number | the model's saved temperature (Flash 1.3302962, 27B 1.2263882), as the official API |
| `raw` | `1`: `{"model", "logits": {<id>: [...]}}`, the raw readout logits of each question's valid options before temperature | off |

Any other query parameter is 400.

### Locate

`{"type": "locate", "instructions": "..."}` extracts verbatim spans of the state. The official model supports it in
the library; the HTTP API does not.

- `instructions` must be a nonempty string over HTTP (the library also accepts the question id as the label).
- The state must be a nonempty string, without images; at most 128 questions per request.
- Locate questions can be mixed with decision questions; the two kinds run in separate forward passes.
- `raw=1` is not available for Locate (400).
- The Locate weights are loaded on the first Locate request (Flash: 3.8 GB more memory, about 0.4 s).

## Python API compared with the official library

| Official `pplx_decider` | decider-mlx |
|---|---|
| `Decider.from_pretrained(checkpoint, device=...)` | `decider_mlx.load(model)`, by official name; downloads the weights when missing and returns once they are mapped |
| `decide(request \| [requests], temperature=)` | `app.decide(request \| [requests], execution_mode="separate" \| "isolated", temperature=None, return_logits=False)` |
| `mode="isolated"` in `from_pretrained` | `execution_mode=` per call |
| `logits(requests)` | `app.decide(..., return_logits=True)` |
| a list raises on the first invalid request | the same for `decide`; `app.decide_each([...])` returns an exception in place of each invalid request |
| `fast=`, `batch_tokens=`, `max_length=`, `locate=`, `warmup()` | none: one configuration per model; graphs are prepared in the background; `app.wait_ready()` waits for that (for measurements) |
| calls serialized by an internal lock | calls from any thread are queued to one worker thread in arrival order |
| `RequestError`, `LengthError` | `decider_mlx.RequestError`, `decider_mlx.LengthError`; `decider_mlx.ModelNameError` for an unknown model, `decider_mlx.fetch.FetchError` for a failed download |

Requests use the library's schema: `{"state", "questions", "images"}`, with images as file paths, data URLs or PIL
images. `decider_mlx` must be imported before `mlx` (it sets `MLX_ENABLE_TF32=0`, which the kernels need).

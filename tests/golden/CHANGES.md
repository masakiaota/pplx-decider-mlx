# Golden changes

The golden outputs were first recorded from the two research engines (the Flash and 27B defaults, MLX 0.32.5).
The release package reproduces every line byte for byte (eval 772, records 772, boolq100 in 4 conditions, and the
HTTP cases), except the HTTP cases below, which changed on purpose. `http.jsonl` of both models was regenerated
for these cases only. Every answer value is unchanged.

| case | before | now | reason |
|---|---|---|---|
| `array_with_bad_item` | 400 for the whole array | 200, an array with `{"error": ...}` in the place of the bad item | an invalid item gets its own error; the other items are answered |
| `alias_latest` | 200 (`pplx-decider-flash-latest` / `pplx-decider-latest` accepted) | 400, `Unknown model ...; this server serves '<name>'` | only the official versioned name is accepted |
| `alias_perplexity_prefix` | 200 (`perplexity/<name>` accepted) | 400, same message | as above |
| `other_model` | 400 `Model '<other>' is not served here` | 400 `Model '<other>' is not served by this server; it serves '<name>'` | the message names the served model |
| `too_long`, `too_long_isolated` | message prefixed `LengthError: ` | the same message without the prefix | error messages no longer carry Python class names |
| `locate_raw` | `RequestError: Raw decision logits do not apply to Locate` | without the prefix | as above |
| `score_1level` | Flash `score needs at least two levels`; 27B `RequestError: Invalid option count` (param null) | both `Question 's': a score needs at least two ordered levels` (param `questions.s.criteria`) | one wording for both models, the official library's |
| `score_11levels` | Flash `score criteria must have at most 10 levels`; 27B `score criteria must have 1 to 10 levels` | both `Question 's': 11 score levels; at most 10 are supported` | one wording for both models (the API allows at most 10 levels) |
| `question_bad_type` | `RequestError: Unsupported question type: yesno` (param null) | `Question 'x': type must be 'choice', 'noul', 'score' or 'locate'` (param `questions.x.type`) | the official library's wording, with Locate |

## Image requests over HTTP read the remaining state as sent (27B)

The 27B HTTP layer used to join the text parts left in `state` after the images were taken out
(`[image, {"type": "text", "text": "..."}]` became the plain text `...`). Flash, and the official library, read
what is left as sent: the remaining array is serialized as JSON. Both models now do the latter, and the per-model
registry flag is gone. `27b/http.jsonl` was regenerated for the 21 cases whose `state` holds images and text parts:

`features16_{separate,isolated}_{12,13,14,15}`, `image_vqa_{0,1,2}`, `image_string_tail`, `image_two`, `image_raw`,
`image_isolated`, `image_noise_png`, `array_mixed10`, `array_mixed10_isolated`, `array_mixed10_saved_temperature`,
`array_mixed10_raw`, `array_mixed80`.

The input token counts of these cases now equal Flash's (for example `image_vqa_0` 458 -> 482), and the answer
values change accordingly. No other line changed; Flash is unchanged.

## Synthetic images and long text in the HTTP cases (both models)

The shipped tests no longer contain VQAv2 / COCO images or the Qasper paper text. In `http-cases.jsonl` and
`http-images.json`:

- The 13 COCO photos (`@image:0` to `8`, `10` to `13`) were replaced by generated JPEGs of the same sizes (coloured
  shapes on a gradient with mild noise), and the VQAv2 questions about them by questions about the new images
  (same question ids, types and option counts). The noise PNG (`@image:9`) and the two invalid data URLs are
  unchanged.
- `long5k`: the Qasper paper text was replaced by a generated station log of the same length (25.2K characters)
  with a question about it.

This changes the inputs, not the engine. `http.jsonl` of both models was regenerated with the same engine for the 21
cases that use these inputs: `features16_{separate,isolated}_{12,13,14,15}`, `image_vqa_{0,1,2}`,
`image_string_tail`, `image_two`, `image_raw`, `image_isolated`, `long5k`, `array_mixed10`,
`array_mixed10_isolated`, `array_mixed10_saved_temperature`, `array_mixed10_raw`, `array_mixed80`. No other line
changed. The generator is kept with the research records (`R-003`).

## Length error message (both models)

The `LengthError` message ended with the official library's advice to "raise max_length", which users of this
engine cannot do: the limit is fixed at 32,768 tokens. The official HTTP API documents no message text for this
case (clients branch on the status and `type`), so the message now states the limit instead. `http.jsonl` of both
models changed in the two cases that hit it, `too_long` and `too_long_isolated`; status, type, code and param are
unchanged:

| before | now |
|---|---|
| `Question 'a' needs 40075 tokens with the state and images, over the 32768-token limit; shorten the state, lower the image resolution or raise max_length` | `Question 'a' needs 40075 tokens with the state and images, which exceeds the 32,768-token input limit; shorten the state or lower the image resolution` |

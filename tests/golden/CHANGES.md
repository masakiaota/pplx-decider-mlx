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

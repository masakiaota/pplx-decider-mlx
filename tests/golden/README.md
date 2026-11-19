# Golden outputs

`test_bitexact.py` compares the engine's outputs with these files byte for byte (JSON with sorted keys and compact
separators). They were recorded on a Mac Studio (M5 Ultra) with MLX 0.32.5 built from source at commit `fc344c9`.
[CHANGES.md](CHANGES.md) lists every line that was regenerated after the first recording, and why.

| File | Contents |
|---|---|
| `boolq100.jsonl` | 100 BoolQ requests (passage, question, `noul` / `choice` / `score` questions) |
| `<model>/boolq-{separate,isolated}-{single,list}.jsonl` | their answers at temperature 1, one request per call and as one list, in both execution modes |
| `http-cases.jsonl` | 104 HTTP cases: every feature of `POST /v1/decisions` and every error path. `@MODEL`, `@OTHER`, `@image:N` and `@repeat:TEXT:N` are expanded by the test |
| `http-images.json` | the images used by the cases, as data URLs |
| `<model>/http.jsonl` | status, headers and body of every case |

## Data sources and licences

- **BoolQ** (Clark et al., 2019, [google-research-datasets/boolean-questions](https://github.com/google-research-datasets/boolean-questions)),
  licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). `boolq100.jsonl` and the BoolQ
  cases in `http-cases.jsonl` contain passages (from Wikipedia) and questions from its validation split, unchanged,
  wrapped into Decider requests. The answer options (`Yes` / `No` / `The passage does not say`, the score levels)
  were written for these tests.
- **SQuAD v2.0** (Rajpurkar et al., 2018, [rajpurkar.github.io/SQuAD-explorer](https://rajpurkar.github.io/SQuAD-explorer/)),
  licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The `locate_*`, `array_mixed*`
  and multi-question cases in `http-cases.jsonl` contain passages (from Wikipedia) and questions from its development
  set, unchanged; the wrong `choice` options were drawn from other answers of the same set.
- **Images**: all images in `http-images.json` are synthetic, generated for these tests (coloured shapes on a
  gradient, a random-noise PNG, and two invalid data URLs). The questions about them were written for these tests.
- Everything else (the synthetic filler text, the museum example, error cases) was written for these tests.

The files under `tests/golden` that contain BoolQ or SQuAD text (`boolq100.jsonl`, `http-cases.jsonl` and the
`http.jsonl` / `boolq-*.jsonl` outputs, which echo option names and Locate spans) are distributed under the
respective CC BY-SA licence of their source. The rest of this repository is Apache-2.0.

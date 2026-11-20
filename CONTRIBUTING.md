# Contributing & Development

This guide covers running tests, benchmarking, and the MLX wheel packaging used by `decider-mlx`.

## Tests

```sh
uv sync                                         # installs dev dependencies including pytest
uv run pytest                                   # model-free tests (API, HTTP validation, fetch via mock Hub)
DECIDER_MLX_TEST_MODELS=pplx-decider-flash-v1.3,pplx-decider-v1.3 HF_HUB_OFFLINE=1 uv run pytest
```

- Setting `DECIDER_MLX_TEST_MODELS` loads the models from the local Hugging Face cache and compares BoolQ under four execution conditions along with every HTTP feature case byte for byte against `tests/golden` (see [tests/golden/README.md](tests/golden/README.md)).
- The golden outputs were recorded on an M5 Ultra; other chips, macOS versions, or MLX builds may differ in the last bits.
- `tests/hub_mock.py` is a local stand-in for the Hugging Face Hub that serves a folder as a model repository, used for testing `fetch` offline.

## Benchmarks & Quality Evaluation

Run in-process speed benchmarks (single requests, a 20-request array, and long inputs):

```sh
uv run python bench/speed.py --model pplx-decider-flash-v1.3
```

To compare a build against the original BF16 models on held-out evaluation sets, see [bench/quality/README.md](bench/quality/README.md).

## About the MLX 0.32.5 Wheels

`uv sync` installs MLX 0.32.5 from the wheels built by MLX's own CI at `ml-explore/mlx` commit `fc344c9`. That revision includes [ml-explore/mlx#4682](https://github.com/ml-explore/mlx/pull/4682) (faster quantized matrix products on M5) and is not yet published on PyPI.

Because CI artifacts expire, the two unmodified wheels (`mlx` and `mlx-metal`, MIT License) are re-hosted on this repository's [`mlx-0.32.5-fc344c9` release](https://github.com/masakiaota/pplx-decider-mlx/releases/tag/mlx-0.32.5-fc344c9), with their sha256 checksums pinned in `uv.lock`.

A local source build produces the same outputs, but runs Flash 5–7% slower on 12.9K-token inputs (other shapes and 27B are within 1%) because MLX's built-in Metal kernels vary with the Xcode toolchain. Once `mlx==0.32.5` is published on PyPI, this project will switch to the official PyPI wheels.

## Architecture, Research Record & Weight Conversion

- [experiments/README.md](experiments/README.md): Research record detailing what was tried for each model, what was kept or dropped, and the measured results (weights, INT8 and W4A8 kernels, length buckets, bulk packing, startup).
- [tools/convert/README.md](tools/convert/README.md): Scripts to rebuild the published MLX weights from the original BF16 checkpoints.

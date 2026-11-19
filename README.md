# decider-mlx

Work in progress. MLX inference engine for Perplexity Decider v1.3 models on Apple silicon Macs.

## Build requirements

`uv sync` builds MLX 0.32.5 from source (ml-explore/mlx commit `fc344c9`, pinned in `pyproject.toml`).
This needs Xcode with the Metal Toolchain (`xcodebuild -showComponent MetalToolchain` should report
`installed`) and network access during the build. The first build takes about 2 minutes on 16 cores;
later syncs reuse the uv cache.

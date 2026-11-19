# Serving

This file covers how requests are grouped into GPU work, how the engine starts, and how processes share the machine. Speed ideas here were held to a strict rule: either outputs stay bit-exact, or the output must not depend on what else is in the batch and quality must be shown equal.

## Length buckets (`S-001`, adopted)

**What.** Short forwards (every chunk up to 512 tokens) are padded to a fixed set of total lengths and run through graphs compiled ahead of time, one graph per bucket and request structure.

**Why.** `mx.compile` traces a new graph for every new total length. For a stream of fresh requests that trace cost (10-15 ms per new shape on Flash, about 0.15 s per graph on 27B) is paid on almost every request.

**How.** The padding is one extra dummy chunk that sees only itself in attention, has its own DeltaNet state and convolution boundary, and has no question or marker, so it never produces output. The layout tensors (chunk starts, owners, question offsets, positions and so on) are graph inputs, so one trace covers every length in a bucket. Buckets are 64 tokens apart up to 512.

**Bit-exactness problem and fix.** The first version matched the unbucketed engine 520/520 with 32-token buckets but only 510/520 with 64 and 376/520 with 128. Adding rows does not change existing rows for INT8 matmul or RMSNorm, but MLX's standard quantized matmul and BF16 matmuls switch kernels at certain row counts (2, 6, 16, 17, 65, 129, 193, 225, 257, 449, 513, ..., 2048 on MLX 0.32.5 / M5), and a different kernel changes every row's bits. Buckets now never cross these boundaries or the engine's own path switches; a length that has no bucket left in its interval takes the old path. After the fix: 151/151 requests, 348/348 forwards at and around every boundary, and 520/520 over HTTP were bit-exact.

**Result.** Flash, 520 random-length requests sent one at a time over HTTP: total 33.2 -> 27.4 s (-17%), median 43.6 -> 33.7 ms, p95 -3% (p95 is set by long text and images, which were not bucketed then). Repeated requests are not slower, unlike the `unique` profile that simply disabled compile. Images were later bucketed the same way (`V-002`). 27B uses 84 bucket graphs for inputs up to 512 tokens.

**Not bucketed: long inputs.** For one long sequence compile is already faster than eager even with the trace (about -15% on Flash), and sharing graphs across long lengths would need an attention kernel that takes the length as an input (`I-001`).

## Bulk packing (`M-001`, adopted for 64+ requests)

**What.** A list of 64 or more requests is tokenized, split by arithmetic domain (INT8, score, Locate), sorted by length and packed first-fit into fixed 2,048-token packs. Leftover space is filled with independent dummy segments, so every pack has the same shape and one compiled graph per domain serves the whole job. Sequences over 512 tokens run one by one on the long-input path. The next pack is assembled while the GPU runs the current one.

**Output contract.** Bulk output is not bit-exact with one-at-a-time processing; neither is the normal list path (279/1,072 identical), because different row counts select different kernels. Instead, bulk output does not depend on how requests are packed: identical for budgets 2,048, 4,096 and 8,192, any input order, a 1-request job vs a 100-request job, and an interrupted and resumed job. Paired difference against one-at-a-time processing on 1,124 questions: mean |dp| +0.0001 [-0.0011, +0.0013]. The gate is 100/100 under all conditions.

**Result.** Flash, 1,072 mixed evaluation requests: 64.8 s with lists of 16 -> 52.2 s (1.24x), later 47.0-47.4 s in the final engine. 27B, 772 requests: 166 s sequential -> 131 s bulk.

**Threshold.** Packing loses on small lists because of pack remainders. Measured on fresh mixed requests, lists of 2-20 were 1.14-2.14x faster on the list path, 32-48 were equal, and 64 and above favoured bulk by 5-11%. "Always bulk for two or more" was rejected; the threshold is 64 (`I-001`, `I-006`).

A separate tokenizer thread for bulk gave nothing and was dropped. Overlapping CPU preparation with GPU work inside normal batches was also dropped earlier: the GPU idles only about 2% of a 20-request batch (`flash-speed/004`).

## Startup

At the start of the daytime round, Flash took 10.8-14.2 s from process start to first answer, against 2.2 s for the official app. Three changes brought it to about 3 s.

**Listen first, prepare between requests (`U-006`, adopted).** The HTTP server binds immediately and returns 503 with `Retry-After: 1` until weights are mapped. After that, warm-up requests, bucket graphs, image graphs and bulk graphs are prepared one step at a time on the inference worker, between requests. Before each step the worker waits 20 ms for a request; without that grace, sequential requests had to wait for a running step (first 30 requests 6.05 s vs 3.25 s with grace). Flash first answer 11.1-11.4 s -> 2.8-3.1 s; 27B 8.29/5.16 s -> 3.31/3.20 s. Answers do not depend on timing: 520/520 bit-exact whether a request arrived during or after preparation.

Trade-offs: under a continuous request stream preparation does not advance (27B finished only after 520 requests, about 139 s after start), and a request with a not-yet-prepared shape traces on the spot (0.1-0.2 s). Right after start, Flash requests take 62-135 ms instead of 46 ms.

For the Python API the release uses the same idea: `load()` returns once weights are mapped and preparation continues on the engine's worker thread between requests; `decide` calls are passed to that thread in FIFO order. In the research engines `load()` blocked until preparation was done (Flash about 14 s, 27B 26.9 s).

**Trace-only preparation (`G-001`, adopted).** Preparation used to trace each graph and then execute it once on synthetic input. Timing per Flash graph: trace about 10 ms (Python 2 ms), Metal JIT about 1.4 ms, execution about 34 ms. Execution dominated, and it is not needed: the compiled graph stays in MLX's compile cache and kernels are JIT-compiled on the first real request. Skipping the execution: preparation complete 12.1 -> 7.1 s (Flash), 22.9 -> 9.3 s (27B, where execution was 13.7 of 15.9 s). First-request latency for new shapes and steady-state speed were unchanged, and 494/494 outputs were bit-exact. Warm-up requests that do run (Flash 2.1 s, 27B 1.4 s) are kept because they also page in weights and compile the custom kernels.

**Graph export (`G-001`, rejected).** `mx.export_function` works for these graphs, including custom Metal kernels, once weights are passed as inputs rather than captured as constants (otherwise each file contains the full weights). 101 Flash bucket graphs exported to 114 MB with 101/101 bit-exact outputs when the imported function is wrapped in `mx.compile` (unwrapped: only 7/101 bit-exact and 17% slower). But importing takes 40 ms per graph against 10 ms to trace, and the bucket stage took 4.02 s with import vs 3.92 s without. MLX 0.32.5 has no persistent Metal pipeline cache, and the JIT share is only about 0.25 s in total. Neither a shipped graph cache nor a first-run cache is used.

**Lazy tokenizer loading (rejected).** Loading the tokenizer first and the processor later made no measurable difference (2.43/2.89 s vs 2.54-2.99 s). The remaining gap to the official app was mostly importing transformers (0.5-1.1 s) and building the processor (1.4 s). The release package removed both (`V-004`): the tokenizer is read with `tokenizers`, the chat template is a fixed string and image preprocessing is two Metal kernels that reproduce the torchvision path byte for byte. With the weights mapped and no blocking warm-up, the server accepts requests 0.3-0.5 s after process start and gives the first answer after 0.51-0.54 s (Flash) and 0.6-1.0 s (27B) (`R-002`).

## Concurrency

**Within one server.** Connections are accepted by a threaded HTTP server and all inference runs on one worker in FIFO order (MLX GPU streams are per thread). This also fixed connection resets under 8 concurrent clients (14/20 -> 0) seen in an earlier server. There is no answer cache.

**Cross-process GPU lock (`P-001`, rejected).** Each model runs as its own `serve` process. A lock that serializes inference across processes was tested against no lock:

| Workload | Sum of solo runs | Concurrent, no lock | Concurrent, lock |
|---|---:|---:|---:|
| Flash + 27B | 47.7 s | 40.4 s (0.85) | 48.0 s (1.01) |
| Flash + Flash | 52.1 s | 38.6 s (0.74) | 51.9 s (1.00) |
| Short Flash + 27B 32K | 37.7 s | 33.8 s (0.90) | 37.7 s (1.00) |

The lock turns concurrency back into serial execution and is 11-35% slower than no lock. Without a lock a second process fills GPU gaps left by host work, synchronization and small kernels. With a lock, a short Flash request can wait the whole 22 s of a 32K 27B request. Outputs were identical in every mode. The two servers are therefore fully independent: no lock, no shared files.

**Two processes on bulk work (`P-002`, informational).** The headroom is smaller for large packs: Flash about 12%, 27B about 4%. Two 27B processes share the memory-mapped weight pages and fit on the test machine. Single-run measurements.

**Two threads in one process (`P-002`, deferred).** Running two Python threads, each with its own GPU stream, was 14% faster for two sequential streams but 10 of 120 outputs differed from serial execution. The engine keeps shared graph and state objects and is single-threaded by design. Making this safe needs per-thread state and a bit-exactness check; not done.

## Memory limits

The research engines set fixed MLX memory limits and checked each request against an estimate before running it. The values changed several times:

- Flash: 20 GB, justified by a function that was never called; set to 16 GB from measurement (32K peak 10.9-13.1 GB). With the Locate sidecar loaded, active memory is 9.55 GB and the preflight estimate (about 0.22 MB per token) exceeded 16 GB, so 32.4K requests, including plain binary ones, failed with MemoryError. The limit went back to 20 GB (measured peak 14.1 GB for 32.4K Locate).
- 27B: 32 GB and 48 GB set in two places, neither measured at 32K. Unified to 40 GB because the preflight estimate for a 32K Locate request was 32.1 GB (measured peak 25.0 GB).

For the release both the fixed limits and the engine's own preflight were removed; the engine uses MLX's defaults, which derive from the Mac's recommended working set. The fixed values had been chosen for one machine, and the preflight estimate itself caused the MemoryError above on requests that actually fit. The 64 GB cap remains only in the benchmark harness. Recommended RAM: 24 GB or more for Flash, 36 GB or more for 27B, based on the measured peaks (Flash 14.2 GB, 27B 25.0 GB at 32K tokens) plus room for the OS.

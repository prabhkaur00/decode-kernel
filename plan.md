# Split KV Decode Attention Kernel

A Triton split KV decode attention kernel, layout compatible with FlashInfer's paged KV cache, characterized against FlashInfer and a naive single block Triton kernel as baselines. Part 1 is the kernel and its characterization. Part 2 is an SGLang integration that runs only on A100 class hardware.

---

## Part 1: Kernel implementation and characterization

### Module 1: Environment and dependencies

**Contents.**
- `env/requirements.txt`: pinned torch, triton, flashinfer, transformers, matplotlib, pandas, numpy.
- `env/setup.sh`: detects CUDA version, installs the matching FlashInfer wheel, verifies Triton compiles a trivial kernel, prints GPU name and compute capability.
- `env/verify.py`: imports every dependency, prints versions, exits non zero on any failure.

### Module 2: KV layout and FlashInfer interop

**Contents.**
- `src/layout.py`: a `PagedKVLayout` dataclass (page size, head dim, num KV heads, dtype); a `synthesize` function that allocates random K, V, Q tensors in FlashInfer's paged layout for a given (batch, context_length, num_q_heads, num_kv_heads, head_dim); a `build_block_table` function that constructs `kv_indptr`, `kv_indices`, `kv_last_page_len` for a uniform context sweep.
- `src/layout.md`: prose description of the exact shape, stride, and dtype of every tensor crossing the kernel boundary. Written from FlashInfer source, not guessed.

### Module 3: Reference attention

**Contents.**
- `src/reference.py`: pure PyTorch fp32 attention on dense Q, K, V; a `gather_paged_to_dense` helper that unpacks the Module 2 layout for the reference to consume.

### Module 4: Naive single block Triton decode kernel

**Contents.**
- `src/kernel_naive.py`: one Triton block per (batch, head group), reads the entire KV inside the block, no split, no reduction pass. GQA broadcast by indexing (`kv_head_idx = q_head_idx // group_size`), not by replicating KV in memory. fp32 accumulators for the softmax statistics. Python wrapper takes paged KV tensors, returns output.

### Module 5: Split KV Triton decode kernel

**Contents.**
- `src/kernel_split_kv.py`: two kernels. Partition pass, parameterized on `SPLIT_KV`, writes per partition partial outputs, row maxes, and row log sum exps to scratch buffers. Reduction pass merges using the online softmax identity:

  ```
  m = max_i(m_i)
  l = sum_i exp(m_i - m) * l_i
  O = sum_i exp(m_i - m) * (l_i / l) * O_i
  ```

  Merge state (m_i, l_i, partial O_i) is fp32 even when inputs are fp16. Python wrapper allocates scratch buffers once and reuses them.

### Module 6: Benchmark harness

**Contents.**
- `bench/microbench.py`: sweeps (implementation, context_length, batch_size, split_kv) and writes a CSV with columns `implementation, context_length, batch_size, split_kv, latency_ms_p50, latency_ms_p95, achieved_bw_gb_s, peak_bw_gb_s, bw_pct_of_peak, max_abs_err_vs_fp32`. CUDA event timing. 20 warmup iterations, 100 timed iterations. L2 cache flushed between iterations by writing through a 256 MB dummy buffer. FlashInfer timed only on its `run` phase, never the `plan` phase. Implementations: flashinfer, naive_triton, split_kv_triton.

### Module 7: Profiling

**Contents.**
- `bench/profile.py`: wraps a single kernel invocation for `ncu` with warmup outside the profiled region. Captures three configurations (short context with small split, long context with optimal split, long context with too many splits) and saves `.ncu-rep` files. Extracts achieved occupancy, DRAM throughput, L2 hit rate, and top two warp stall reasons into a CSV.

### Module 8: End to end smoke test

**Contents.**
- `bench/e2e_smoke.py`: loads Llama 3.2 1B, replaces the attention forward in one decoder layer with the split KV kernel, runs short generation on a handful of hardcoded prompts, asserts top 5 token agreement on the next predicted token between stock and patched models. Not timed. Sanity check only.

### Module 9: Writeup

**Contents.**
- `writeup/REPORT.md`: setup, kernel design with the online softmax merge math, correctness summary, latency results with the U shaped curves, roofline analysis, comparison vs FlashInfer with an honest decomposition of the gap, limitations.

---

## Part 2: SGLang integration (A100 only)

Do not begin this until Part 1 is complete and all correctness tests pass. SGLang requires Ampere class hardware (compute capability 8.0 or higher), and the `sgl_kernel` prebuilt wheels do not always include sm_80 binaries, so plan for a source build of `sgl_kernel`.

### Module 10: SGLang attention backend

**Contents.**
- `integration/sglang_backend.py`: a subclass implementing SGLang's attention backend interface. Forwards to the Module 5 kernel using the paged KV layout SGLang already passes in (FlashInfer compatible by default). A registration hook that selects this backend via an environment variable or programmatic config.
- `integration/README.md`: source build instructions for `sgl_kernel` on sm_80, the exact SGLang commit pinned, the launch command for the server subprocess on Colab, and the port to ping.

### Module 11: End to end benchmark

**Contents.**
- `integration/bench_e2e.py`: launches an SGLang server with the custom backend on Llama 3.2 1B, runs SGLang's own throughput benchmark script with a fixed prompt mix, captures tokens per second, time to first token, and time between tokens. Compares against SGLang with the stock FlashInfer backend on the same workload.
- Results written to `results/sglang/`.

---

## How to ensure the implementation is correct

Correctness is gated at three points and the benchmark harness refuses to run if any gate fails.

**Gate 1: layout sanity, before any custom kernel exists.** Use Module 2 to synthesize paged tensors, run them through FlashInfer's decode wrapper, and compare against the Module 3 reference on the dense form of the same data. If FlashInfer disagrees with the reference, the block table or layout is wrong, not the kernels. This catches the most common silent bug, which is a malformed `kv_indices` tensor that produces plausible looking garbage.

**Gate 2: kernel correctness on the full sweep grid.** For every (implementation, context_length, batch_size, split_kv) combination in the benchmark, compare against the fp32 reference before timing. Tolerances: max absolute error below 1e-2 for fp16, below 1e-3 for fp32. Also assert mean absolute error below 1e-3 for fp16; the mean check catches systematic bias that the max check tolerates. The naive kernel and the split KV kernel must both pass; if the split KV kernel disagrees with the naive kernel even when both pass the fp32 check, that is also a signal worth investigating (numerical drift in the merge).

**Gate 3: edge cases and determinism.** Context length 1, context length not divisible by page size, context length not divisible by `SPLIT_KV` (forces ragged partitions), batch 1, GQA group size 1 (which is MHA), GQA group size equal to num Q heads (which is MQA). Run each kernel twice on identical inputs and assert bitwise identical outputs; non determinism here points to a race in the reduction pass.

The most common silent bug is storing the merge state (m_i, l_i, partial O_i) in fp16 instead of fp32. The error scales with the number of splits and does not show up at short context. If max error grows with `SPLIT_KV` at fixed context, check the dtype of the scratch buffers first.

---

## What to vary in ablations

Five axes, in priority order.

**SPLIT_KV at fixed context length.** This is the headline experiment. Sweep `(1, 2, 4, 8, 16)` at each of `(2k, 8k, 32k, 64k)`. Expected: U shaped latency curve at each context length, with the minimum shifting to larger `SPLIT_KV` as context grows.

**Context length at fixed split.** Sweep `(2k, 4k, 8k, 16k, 32k, 64k)` for each implementation at `SPLIT_KV = 1` and at the per length optimal `SPLIT_KV`. Expected: naive scales worse than split KV at long context, FlashInfer scales best.

**Batch size.** Sweep `(1, 4, 16)` at fixed long context (32k). Expected: split KV advantage shrinks as batch grows, because batch already provides parallelism that split KV was synthesizing.

**BLOCK_N (KV block size inside a partition).** Sweep `(32, 64, 128)` at the per length optimal `SPLIT_KV`. Expected: 64 is usually best on Ampere; 128 may win at very long context due to better amortization of address arithmetic.

**GQA group size.** Compare native group size of the chosen model against an artificially expanded version (replicate KV heads to simulate MHA). Expected: smaller group sizes (closer to MHA) cost more bandwidth and run slower; this confirms the GQA broadcast in the kernel is doing what it should.

---

## What plots to look at and what they reveal

Six plots, all generated from the Module 6 CSV.

**Plot 1: latency vs SPLIT_KV, one panel per context length, one line per implementation.** Should produce the U shape for split KV. FlashInfer appears as a horizontal reference line (it does not expose `SPLIT_KV` to the user). Naive Triton appears at `SPLIT_KV = 1`. Reading this plot: the location of the minimum tells you the optimal split for that context length on this GPU.

**Plot 2: latency vs context length, one line per implementation, at each implementation's optimal split.** Reading this plot: how the three implementations scale. The slope tells you which is most context limited.

**Plot 3: achieved bandwidth as percent of peak vs context length, one line per implementation.** Reading this plot: how close to the memory bound each implementation gets. A flat horizontal line near peak says the kernel is fully bandwidth bound and there is no headroom from kernel work alone. A line that climbs with context length says the kernel is amortizing fixed overheads as work grows.

**Plot 4: roofline.** Log log axes. X is arithmetic intensity in FLOPs per byte. Y is achieved performance in TFLOPs. The GPU's peak compute and peak HBM bandwidth define the two ceilings. Each implementation is a single point per configuration. Reading this plot: decode attention sits firmly in the memory bound region; the only way to move right (higher arithmetic intensity) is to batch more or to share KV across more queries (which is what cascade attention does). This plot is the single most important interview talking point.

**Plot 5: speedup heatmap, split KV over naive.** Rows are context lengths, columns are batch sizes, cell color is the ratio. Reading this plot: where split KV pays off and where it does not. Top right corner (long context, small batch) is where the speedup is largest.

**Plot 6: gap to FlashInfer, split KV at optimal SPLIT_KV.** Bar chart, one bar per context length, height is `latency(split_kv) / latency(flashinfer)`. Reading this plot: where your implementation closes the gap and where FlashInfer pulls ahead. A widening gap at long context typically indicates persistent kernel design and autotuned tile sizes that Triton cannot match.

The writeup in Module 9 must reference each of these plots by number and explain what it shows.
````
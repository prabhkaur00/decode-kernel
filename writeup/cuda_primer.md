# CUDA Primer — Reading the CUDA Kernels in This Repo

A bottom-up introduction to CUDA C++ for someone who knows PyTorch but has
never written a GPU kernel.  By the end you should be able to read every line
in `src/cuda/attention_ext.cu` and `src/cuda/decode_attn.cuh` without
confusion.

---

## 1  The CUDA programming model

In CUDA you write a kernel function that runs once per **thread**.  You decide
which piece of work each thread owns by reading three built-in variables:

```cpp
threadIdx.x   // thread's index within its block   (0 … blockDim.x-1)
blockIdx.x    // block's index within the grid      (0 … gridDim.x-1)
blockDim.x    // number of threads per block
```

The standard pattern for a 1-D workload:

```cpp
__global__ void add_kernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// Launch: ceil(N/256) blocks, 256 threads each
add_kernel<<<(N+255)/256, 256>>>(A, B, C, N);
```

For multi-dimensional grids (used in decode attention):

```cpp
dim3 grid(batch, num_q_heads);   // 2-D grid
dim3 block(head_dim);            // 1-D block: head_dim threads per block

kernel<<<grid, block>>>(args...);

// Inside the kernel:
int batch_idx  = blockIdx.x;
int q_head_idx = blockIdx.y;
int tid        = threadIdx.x;    // head-dimension index (0…head_dim-1)
```

One CUDA block per (batch, query-head) pair.  One thread per head dimension.

---

## 2  Thread hierarchy — the four levels

```
Grid
├── Block (0,0)  ← CTA (cooperative thread array), runs on one SM
│   ├── Warp 0  (threads  0–31)  ← 32 threads that execute in lockstep
│   ├── Warp 1  (threads 32–63)
│   └── ...      (up to 32 warps = 1024 threads per block)
├── Block (1,0)
└── ...
```

Key relationships:

| Level | Size | Communication |
|-------|------|---------------|
| Thread | 1 | Registers only |
| Warp | 32 | Registers + shuffle instructions (`__shfl_*`) |
| Block | ≤1024 | Registers + shared memory + `__syncthreads()` |
| Grid | unlimited | Global memory only |

Within a warp, all 32 threads execute **the same instruction each cycle**
(SIMT — Single Instruction, Multiple Threads).  Divergent branches
(e.g., `if (tid < 16)`) serialize the warp: the two paths run one at a time.

---

## 3  The three memory spaces you control

### 3.1  Registers

Every local variable in a CUDA kernel lives in a register by default.

```cpp
float q_val = ...;      // register: ultra-fast, private to this thread
float acc   = 0.0f;     // register accumulator
float parts[PAGE_SIZE]; // small fixed-size array: lives in registers
```

Register pressure matters: if a kernel uses too many registers, the compiler
"spills" excess values to **local memory** — which physically lives in L2/HBM
and is very slow.  `nvcc --ptxas-options=-v` reports spills.

### 3.2  Shared memory (SRAM)

On-chip memory shared by all threads **in the same block**.  ~164 KB per SM on
A100, partitioned among resident blocks.

Declared with `__shared__` (compile-time size) or `extern __shared__` (size
passed at launch):

```cpp
// Static allocation
__shared__ float smem_k[16 * 128];   // PAGE_SIZE × HEAD_DIM

// Dynamic allocation — size provided in the <<<..., smem_bytes>>> argument
extern __shared__ float smem[];
float* smem_k = smem;                         // first region
float* smem_v = smem + PAGE_SIZE * HEAD_DIM;  // second region
```

Shared memory is the primary tool for **inter-thread communication within a
block**.  The decode attention kernel uses it to:

1. Stage a K or V tile (PAGE_SIZE × HEAD_DIM floats) from HBM
2. Hold per-warp partial QK sums before the cross-warp reduction
3. Hold the final PAGE_SIZE QK scores before the softmax computation

You must call `__syncthreads()` after writing shared memory and before reading
it from another thread (see §4).

### 3.3  Global memory (HBM)

All PyTorch tensors live here.  Every `tensor.data_ptr()` is a pointer into
HBM.  Accesses are the bottleneck: ~2 TB/s on A100 vs ~20 TB/s for shared
memory.  The goal is to minimise HBM round trips by staging data through shared
memory or keeping it in registers.

---

## 4  Synchronisation

### 4.1  `__syncthreads()`

Waits until **every thread in the block** has reached this statement.
Required before reading shared memory that another thread has written.

```cpp
// Thread 0 writes, all threads read — BROKEN without sync:
if (threadIdx.x == 0) smem[0] = compute();
float x = smem[0];             // data race: other threads may see stale value

// CORRECT:
if (threadIdx.x == 0) smem[0] = compute();
__syncthreads();               // barrier: all threads pause here
float x = smem[0];             // guaranteed fresh value
```

The naive kernel issues **4 syncs per KV page**:

| Sync | After writing | Before reading |
|------|--------------|----------------|
| 1 | K tile to `smem_kv` | QK dot-product reads `smem_kv` |
| 2 | Warp partial sums to `smem_warp` | Cross-warp reduction reads `smem_warp` |
| 3 | Final QK scores to `smem_qk` | Softmax stat computation reads `smem_qk` |
| 4 | V tile to `smem_kv` | Accumulator update reads `smem_kv` |

`__syncthreads()` is expensive only if threads arrive at the barrier at very
different times (serialisation).  When all threads do roughly equal work per
iteration — as in the attention loop — the cost is a few dozen cycles.

### 4.2  Warp-level shuffle — `__shfl_xor_sync`

Threads within the same warp can exchange **register values directly**,
without shared memory and without a sync:

```cpp
__shfl_xor_sync(mask, val, delta)
```

Exchanges `val` between the thread and the thread whose lane ID differs by
`delta`.  A butterfly XOR pattern (delta = 16, 8, 4, 2, 1) implements a full
warp reduction in 5 instructions:

```
Initial:  lane 0 has a0, lane 1 has a1, …, lane 31 has a31

delta=16: lane 0 ↔ lane 16, lane 1 ↔ lane 17, …  (each lane adds its partner)
delta=8:  lane 0 ↔ lane 8,  lane 1 ↔ lane 9,  …
delta=4:  lane 0 ↔ lane 4,  …
delta=2:  lane 0 ↔ lane 2,  …
delta=1:  lane 0 ↔ lane 1,  …

After step 5: every lane holds  a0 + a1 + … + a31
```

This is `warp_reduce_sum` in `decode_attn.cuh`:

```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, delta);
    return val;
}
```

The `0xffffffff` mask means all 32 lanes participate.

---

## 5  Coalesced memory access

When 32 threads in a warp issue global memory loads simultaneously, the
memory controller tries to **merge** them into as few transactions as possible.

```
Coalesced (1 transaction):
  Thread  0 reads addr 1000
  Thread  1 reads addr 1002    → 64 consecutive bytes → 1 cache-line fetch
  Thread  2 reads addr 1004
  ...
  Thread 31 reads addr 1062

Uncoalesced (32 transactions):
  Thread  0 reads addr    0
  Thread  1 reads addr 2048    → each is a separate 128-byte cache line
  Thread  2 reads addr 4096       even though only 2 bytes are needed
```

Uncoalesced access wastes 63 out of every 64 bytes fetched — a 32× bandwidth
penalty.

In the kernel, thread `tid` always accesses `KV[... + tid]`.  Consecutive
threads access consecutive addresses — coalesced.  For fp16 elements (2 bytes
each), one warp fetches 32 × 2 = 64 bytes in a single transaction.

Ensuring coalescing requires deliberately mapping `threadIdx.x` to the
innermost (fastest-varying) dimension of your data.  The decode attention
kernel maps `tid` to the head dimension `d`, which is the innermost dimension
of `kv_data` (stride = 1).

---

## 6  Template parameters as compile-time constants

```cpp
template<int HEAD_DIM, int PAGE_SIZE>
__global__ void decode_attn_naive_kernel(...) {
    constexpr int NUM_WARPS = HEAD_DIM / 32;  // resolved at compile time
    float parts[PAGE_SIZE];                    // fixed-size register array
    #pragma unroll
    for (int tok = 0; tok < PAGE_SIZE; tok++) ...
```

`template<int HEAD_DIM, int PAGE_SIZE>` instructs the compiler to generate a
separate binary for each value combination.  With compile-time trip counts:

- `#pragma unroll` fully unrolls the loop — no branch instructions
- Array sizes are known, enabling optimal register allocation
- Arithmetic like `HEAD_DIM / 32` folds to a literal at compile time

The Python binding dispatches to the right specialisation at runtime:

```cpp
if      (head_dim ==  64) launch_naive_hd< 64>(args...);
else if (head_dim == 128) launch_naive_hd<128>(args...);
else if (head_dim == 256) launch_naive_hd<256>(args...);
```

For an unsupported value, `TORCH_CHECK(false, ...)` raises a Python exception.

---

## 7  The QK reduction — step by step

Each thread owns one head dimension `d` and must compute the full dot product
`sum_d q[d] * K[tok][d]` for every token `tok` in the page.  This requires
summing across all HEAD_DIM=128 threads (4 warps).

### Step A: partial products into registers

```cpp
float parts[PAGE_SIZE];
#pragma unroll
for (int tok = 0; tok < PAGE_SIZE; tok++)
    parts[tok] = q_val * sm.kv[tok * HEAD_DIM + tid];
// parts[tok] = q[tid] * K[tok][tid] — one of 128 summands
```

### Step B: warp-level reduce (no shared memory, no sync)

```cpp
#pragma unroll
for (int tok = 0; tok < PAGE_SIZE; tok++)
    parts[tok] = warp_reduce_sum(parts[tok]);
// After: every lane in the warp holds the sum over 32 dimensions
```

For HEAD_DIM=128 there are 4 warps.  Each warp now holds its 32-dimension
partial sum — but the 4 partial sums still need to be combined.

### Step C: write warp sums to shared memory

```cpp
if (lane_id == 0) {
    #pragma unroll
    for (int tok = 0; tok < PAGE_SIZE; tok++)
        sm.warp[warp_id * PAGE_SIZE + tok] = parts[tok];
}
__syncthreads();   // sync 2: all warp sums visible
```

Only lane 0 of each warp writes (all lanes hold the same value after the
butterfly reduce, so this is not a race).

### Step D: first PAGE_SIZE threads sum across warps

```cpp
if (tid < PAGE_SIZE) {             // threads 0–15, all in warp 0
    float score = 0.0f;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; w++)
        score += sm.warp[w * PAGE_SIZE + tid];
    sm.qk[tid] = (tid < valid) ? score : -1e9f;
}
__syncthreads();   // sync 3: QK scores finalised
```

Total cost: 2 `__syncthreads()` calls and 4 × PAGE_SIZE warp-sum writes/reads
to produce 16 dot-product scores per page iteration.

---

## 8  `__half` — half-precision in CUDA

PyTorch stores fp16 tensors as contiguous arrays of `__half` values.  All
arithmetic in the kernel is fp32 (register-width), so you convert at the
memory boundary:

```cpp
// Load fp16 from global memory → fp32 register
float k_val = __half2float(KV[page * stride_kvp + ... + tid]);

// Store fp32 register → fp16 global memory
O[batch * stride_ob + head * stride_oh + tid] = __float2half(acc / l_i);
```

Both intrinsics compile to single hardware instructions (`cvt.f32.f16` and
`cvt.rn.f16.f32`).  The conversion cost is negligible compared to the HBM
access time.

To get an `__half*` from a PyTorch tensor in the extension binding:

```cpp
const __half* q_ptr = reinterpret_cast<const __half*>(q.data_ptr());
```

This cast is safe because `torch::kFloat16` and `__half` share the same
binary representation (IEEE 754 binary16).

---

## 9  Advanced CUDA techniques (used by production kernels)

The kernel in this repo is intentionally educational — readable and correct,
but not maximally optimised.  Production attention kernels (FlashAttention,
FlashInfer) add:

### 9.1  Persistent kernels

A kernel can loop internally over many tiles without exiting:

```cpp
while (true) {
    int tile = atomicAdd(&global_work_counter, 1);
    if (tile >= total_tiles) break;
    process_tile(tile);
}
```

Benefits: amortises kernel launch overhead (~5–10 µs per launch) and enables
fine-grained dynamic load balancing across SMs.

### 9.2  Software pipelining (async memory copies)

CUDA 11.x introduced `cuda::pipeline` and `cp.async` PTX to overlap HBM loads
with arithmetic, hiding memory latency:

```cpp
// Start loading tile N+1 to smem_next
__pipeline_memcpy_async(smem_next, gmem_next, bytes);
__pipeline_commit();

// Compute on tile N (already in smem_current)
compute(smem_current);

// Wait for tile N+1 to arrive
__pipeline_wait_prior(0);
swap(smem_current, smem_next);
```

FlashInfer typically pipelines 2 stages (double-buffering), recovering ~20–30%
performance vs non-pipelined kernels by keeping the arithmetic units busy while
HBM loads are in flight.

### 9.3  Warp specialisation

Different warps within a block can perform different roles simultaneously —
a "producer" warp issues async loads while "consumer" warps compute.
Coordination uses per-warp barriers (`__bar_arrive`, `__bar_wait`), which are
finer-grained than `__syncthreads()` (which stalls every warp in the block).

### 9.4  Tensor Cores (WMMA / MMA)

For prefill (Q sequence length > 1), the QK computation is a matrix-matrix
product.  Tensor Cores accelerate this using dedicated hardware:

```cpp
// 16×16×16 fp16 matrix-multiply-accumulate in a single warp instruction
nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
```

For decode (Q sequence length = 1), QK is a matrix-vector product and Tensor
Cores help less.  The V accumulation can still benefit from them in
high-batch scenarios.

---

## 10  Reading `attention_ext.cu`

Suggested reading order:

| Section | Lines | What to look for |
|---------|-------|-----------------|
| `decode_attn.cuh` | 1–60 | `warp_reduce_sum` butterfly; `SmemLayout` sub-allocations |
| `decode_attn_naive_kernel` | 80–200 | The 4 syncs; where shared memory is written vs read |
| `decode_attn_partition_kernel` | 200–300 | Two differences from naive: `blockIdx.z = split_idx`, output to scratch |
| `decode_attn_reduce_kernel` | 300–340 | Why no shared memory: all threads redundantly compute the same scalars |
| `decode_attention_naive_cuda` | 340–380 | `reinterpret_cast<__half*>`, head_dim dispatch to template specialisations |
| `build.py` | all | `torch.utils.cpp_extension.load`, `-gencode` flag for GPU arch |

---

## 11  Glossary

| Term | Meaning |
|------|---------|
| CTA | Cooperative Thread Array — a block of up to 1024 threads that run on one SM |
| SM | Streaming Multiprocessor — the GPU's execution unit; A100 has 108 |
| warp | 32 threads that execute one instruction per cycle (SIMT) |
| lane | one thread within a warp; `lane_id = threadIdx.x % 32` |
| `__shfl_xor_sync` | register exchange between lanes in a warp; no shared memory, no sync needed |
| `__syncthreads()` | block-wide barrier; all threads in the block must reach it before any proceed |
| bank conflict | two threads in a warp accessing the same shared memory bank; requests serialise |
| coalesced access | consecutive threads read consecutive global addresses; single memory transaction |
| occupancy | fraction of maximum resident warps on an SM; low = SM under-utilised |
| register spill | excess registers written to L2/HBM ("local memory"); very slow |
| persistent kernel | a kernel that loops internally over many tiles without exiting and re-launching |
| `cp.async` | PTX instruction: copies HBM → shared memory without stalling the warp |
| template parameter | C++ compile-time integer constant; enables loop unrolling and fixed-size arrays |
| `__half` | CUDA's 16-bit float type; load/store with `__half2float` / `__float2half` |
| `extern __shared__` | dynamically-sized shared memory; size passed at launch as the third `<<<>>>` arg |

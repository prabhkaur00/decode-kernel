# CUDA Kernel Study Guide: Decode Attention

This is an incremental walkthrough. We start with the problem, define a concrete example, and trace every byte of data through the kernel — introducing CUDA concepts exactly when they appear in the pipeline.

---

## The Problem

At LLM inference time, after generating token `t`, we need to attend the new query vector `Q` over the entire KV cache (all past tokens). The result is:

```
output = softmax(Q · Kᵀ / sqrt(head_dim)) · V
```

The KV cache is stored in **pages** — fixed-size chunks of tokens — because sequences grow dynamically and contiguous allocation would waste memory. Our kernel reads pages from GPU memory (HBM), computes attention scores, and writes one output vector.

---

## The Running Example

We'll track this exact setup throughout:

| Property | Value |
|---|---|
| Batch size | 1 sequence |
| Heads | 1 query head, 1 KV head |
| `head_dim` | 4 (one thread per dimension) |
| `page_size` | 2 tokens per page |
| Sequence length | 4 tokens = 2 pages |

```
Query:  q = [1.0, 1.0, 0.0, 0.0]   ← stored as fp16 in HBM

Page 0 — tokens 0 and 1:
  K[0] = [1, 0, 1, 0]    V[0] = [0.5, 0.5, 0.5, 0.5]
  K[1] = [0, 1, 0, 1]    V[1] = [0.5, 0.5, 0.5, 0.5]

Page 1 — tokens 2 and 3:
  K[2] = [1, 0, 0, 1]    V[2] = [0.5, 0.5, 0.5, 0.5]
  K[3] = [0, 1, 1, 0]    V[3] = [0.5, 0.5, 0.5, 0.5]
```

Expected result: all dot products come out equal, so softmax gives uniform weights → output = `[0.5, 0.5, 0.5, 0.5]`.

---

## Thread-to-Work Mapping

In CUDA, one **kernel** function runs once per **thread**. Threads are grouped into **blocks**; within a block, groups of 32 threads form a **warp** that executes in lockstep.

CUDA exposes three built-in read-only variables inside every kernel:
- `threadIdx.x` — this thread's index within its block (0 … blockDim.x−1)
- `blockIdx.x`, `blockIdx.y` — this block's position in the 2-D grid
- `blockDim.x` — number of threads per block (equals `HEAD_DIM` here)

For this kernel the mapping is rigid:

```
Grid:  one block per (batch_item, query_head)   → launched as dim3(B, G·K)
Block: head_dim threads                          → launched as dim3(D)

→ In our example: 1 block total, 4 threads (one per dimension)
```

Inside the kernel, all per-thread identity variables are derived from those built-ins:
```cpp
int batch_idx   = blockIdx.x;    // which sequence in the batch this block handles
int q_head_idx  = blockIdx.y;    // which query head this block handles
int tid         = threadIdx.x;   // which head-dim position this thread owns (0 … D−1)
int warp_id     = tid / 32;      // which warp within the block (0 … NUM_WARPS−1)
int lane_id     = tid % 32;      // position of this thread within its warp (0 … 31)
```

Thread 0 owns `d=0`, thread 1 owns `d=1`, etc. **Every thread runs the same code but on its own dimension.**

---

## The Naive Kernel

### Step 0: Launch — Template Dispatch

The host (Python/C++) calls:
```cpp
decode_attention_naive_cuda(q, kv_data, kv_indptr, kv_indices, kv_last_page_len)
```

The launcher inspects `head_dim` at runtime and picks the right compiled binary:
```cpp
if      (head_dim ==  64) launch_naive_hd< 64>(args...);
else if (head_dim == 128) launch_naive_hd<128>(args...);
else if (head_dim == 256) launch_naive_hd<256>(args...);
```

Each path calls a different template instantiation:
```cpp
template<int HEAD_DIM, int PAGE_SIZE>
__global__ void decode_attn_naive_kernel(...) { ... }
```

This matters because the kernel uses those integers to define array sizes and loop counts at compile time.

---

> ### Concept: `constexpr`
>
> `constexpr` tells the compiler: "evaluate this at compile time, not at runtime."
>
> **Custom example:**
> ```cpp
> template<int N>
> __global__ void halve_kernel(float* data) {
>     constexpr int HALF = N / 2;   // computed when nvcc compiles this template
>     // HALF is now a literal integer in the binary — no division at runtime
>     data[threadIdx.x] += data[threadIdx.x + HALF];
> }
> ```
> Without `constexpr`, the compiler might not know `N/2` is a fixed constant and can't use it for array sizes or loop bounds.
>
> In our kernel:
> ```cpp
> constexpr int NUM_WARPS = HEAD_DIM / 32;  // e.g. 128/32 = 4, baked in at compile time
> float parts[PAGE_SIZE];                    // array size must be a compile-time constant
> ```
> With `head_dim=4` in our example: `NUM_WARPS = 4/32 = 0` in integer division (that's fine — it means 1 warp handles everything). In the real kernel with `HEAD_DIM=128`: `NUM_WARPS = 4`.

---

### Step 1: Load the Query Into Registers

```cpp
const float q_val =
    __half2float(Q[batch_idx * stride_qb + q_head_idx * stride_qh + tid]) * scale;
```

- `Q` — the query tensor in HBM, stored as fp16 (`__half`)
- `batch_idx`, `q_head_idx`, `tid` — derived from CUDA built-ins in the section above
- `stride_qb`, `stride_qh` — how many elements to skip per batch item / per query head (explained below)
- `__half2float(x)` — converts one fp16 value to fp32; all arithmetic in the kernel stays in fp32 registers
- `scale` — `1 / sqrt(D)`, computed once on the host and passed as a kernel argument; multiplying here folds the scaling into the query load so it doesn't need to be applied again later

**Why strides?** A PyTorch tensor is a flat 1D array in HBM. The tensor `Q` has shape `(B, G·K, D)` — where `B` is batch size, `K` is number of KV heads, `G` is the group factor (query heads per KV head), and `D` is head_dim — but memory is just one long block of fp16 values. Note there is no sequence-length dimension: at decode time we process exactly one new query token per step, so that dimension is 1 and dropped. Strides tell you how many *elements* to skip per step in each dimension.

`KV` has shape `(B, K, D)`. Because this kernel supports grouped-query attention (GQA), `G·K` query heads share `K` KV heads, and `kv_head_idx = q_head_idx / G` maps each query head to its KV head.

For a contiguous `Q` of shape `(B=2, G·K=4, D=4)`:
```
Memory layout (flat):
  index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 ...
         [b=0,h=0,d=0..3] [b=0,h=1,d=0..3] [b=0,h=2,d=0..3] ...

stride_qb = G·K * D = 4 * 4 = 16   (skip 16 elements to reach the next batch item)
stride_qh = D       = 4             (skip 4 elements to reach the next query head)
           (d has stride 1 — adjacent dims are adjacent in memory, no multiplication)
```

So element `Q[b, h, d]` lives at flat index:
```
b * stride_qb  +  h * stride_qh  +  d
```

In the kernel, `tid` *is* `d` — thread 0 owns `d=0`, thread 1 owns `d=1`, etc. So each thread plugs in its own `tid` and reads exactly one fp16 value. The host passes the actual strides from PyTorch (via `q.stride(0)` and `q.stride(1)`), which handles non-contiguous tensors automatically without changing any kernel code.

In our example (`B=1, G·K=1, D=4`, batch_idx=0, q_head_idx=0):
```
Thread 0 reads Q[0 * stride_qb + 0 * stride_qh + 0] = Q[0] = 1.0
Thread 1 reads Q[0 * stride_qb + 0 * stride_qh + 1] = Q[1] = 1.0
Thread 2 reads Q[0 * stride_qb + 0 * stride_qh + 2] = Q[2] = 0.0
Thread 3 reads Q[0 * stride_qb + 0 * stride_qh + 3] = Q[3] = 0.0
```

Each result is then multiplied by `scale = 1/sqrt(head_dim) = 0.5` and stored in a register that **never leaves** — it's reused in every page iteration.

After this step:
```
Thread 0: q_val = 0.5
Thread 1: q_val = 0.5
Thread 2: q_val = 0.0
Thread 3: q_val = 0.0
```

---

### Step 2: Shared Memory Layout

Before the page loop, shared memory is carved into three non-overlapping regions by `SmemLayout`:

```
smem_kv   [PAGE_SIZE × HEAD_DIM floats]  — K or V tile (reused between K and V loads)
smem_warp [NUM_WARPS × PAGE_SIZE floats] — each warp's partial QK sum
smem_qk   [PAGE_SIZE floats]             — final dot-product scores
```

For our example (HEAD_DIM=4, PAGE_SIZE=2, NUM_WARPS=1):
```
smem_kv:   8 floats  (2 tokens × 4 dims)
smem_warp: 2 floats  (1 warp × 2 tokens)
smem_qk:   2 floats  (2 scores)
Total:    12 floats = 48 bytes
```

---

### Step 3: Page Loop — Load K Into Shared Memory

For each page, the kernel first brings the K tile from HBM into shared memory. Thread `tid` loads `K[tok][tid]` for every token in the page:

```cpp
for (int tok = 0; tok < PAGE_SIZE; tok++) {
    sm.kv[tok * HEAD_DIM + tid] = __half2float(k_base[tok * stride_kvs + tid]);
}
__syncthreads();  // ← sync 1: ensure all threads see the full K tile
```

After loading **Page 0** in our example, `smem_kv` holds:
```
index:       0    1    2    3    4    5    6    7
             [K[0][0] K[0][1] K[0][2] K[0][3] K[1][0] K[1][1] K[1][2] K[1][3]]
value:       [  1.0    0.0    1.0    0.0    0.0    1.0    0.0    1.0  ]
```

Consecutive threads load consecutive addresses (`tid` = innermost index), so all 4 loads happen in a single memory transaction. Without `__syncthreads()`, thread 0 might read `smem_kv[5]` (written by thread 1) before thread 1 has written it.

---

### Step 4: QK Dot Products — Partial Products and Warp Reduce

Each thread computes its contribution to the dot product for every token:

```cpp
float parts[PAGE_SIZE];
#pragma unroll
for (int tok = 0; tok < PAGE_SIZE; tok++)
    parts[tok] = warp_reduce_sum(q_val * sm.kv[tok * HEAD_DIM + tid]);
```

---

> ### Concept: `#pragma unroll`
>
> A loop with a compile-time trip count can be **unrolled**: the compiler writes out every iteration as straight-line code, eliminating the loop counter and branch instructions.
>
> **Custom example:**
> ```cpp
> // With unroll (PAGE_SIZE=2 is a compile-time constant):
> #pragma unroll
> for (int i = 0; i < 2; i++)
>     acc += data[i];
>
> // Compiler emits exactly this — no counter, no branch:
> acc += data[0];
> acc += data[1];
> ```
> This matters on GPUs because branches disrupt the SIMT pipeline. `#pragma unroll` works here because `PAGE_SIZE` is a template parameter (a compile-time constant), so the compiler knows the trip count.

---

Before the warp reduce, each thread holds its own dimension's partial product:

```
Thread 0: q_val * K[tok=0][d=0] = 0.5 * 1.0 = 0.5  → parts[0]
Thread 1: q_val * K[tok=0][d=1] = 0.5 * 0.0 = 0.0  → parts[0]
Thread 2: q_val * K[tok=0][d=2] = 0.0 * 1.0 = 0.0  → parts[0]
Thread 3: q_val * K[tok=0][d=3] = 0.0 * 0.0 = 0.0  → parts[0]
```

The dot product `q · K[0]` is the **sum** of these four partials = 0.5. That sum is computed by `warp_reduce_sum`.

---

> ### Concept: `__shfl_xor_sync`
>
> Threads in the same warp can swap register values directly — no shared memory, no sync needed.
>
> `__shfl_xor_sync(mask, val, delta)` exchanges `val` between any two threads whose lane IDs differ by XOR-ing with `delta`.
>
> **Custom example** — 4 threads in a warp, each with a value, want the total sum:
> ```
> Initial:   Thread 0: 3   Thread 1: 1   Thread 2: 4   Thread 3: 2
>
> delta=2 (lane i swaps with lane i^2):
>   T0 ↔ T2: 3+4=7    T1 ↔ T3: 1+2=3
>   After:   T0: 7    T1: 3    T2: 7    T3: 3
>
> delta=1 (lane i swaps with lane i^1):
>   T0 ↔ T1: 7+3=10   T2 ↔ T3: 7+3=10
>   After:   T0: 10   T1: 10   T2: 10   T3: 10  ✓
> ```
> Every thread ends up holding the total sum — this is a **broadcast reduce** in 2 steps (log₂4).
>
> For a full 32-thread warp, the pattern runs 5 steps (delta = 16, 8, 4, 2, 1):
> ```cpp
> __device__ __forceinline__ float warp_reduce_sum(float val) {
>     #pragma unroll
>     for (int delta = 16; delta > 0; delta >>= 1)
>         val += __shfl_xor_sync(0xffffffff, val, delta);
>     return val;  // every lane holds the warp-wide sum
> }
> ```
> `0xffffffff` is a bitmask saying all 32 lanes participate.

---

After `warp_reduce_sum`, every thread in the warp holds the same value: the sum of partial products across all warp lanes.

In our example (all 4 threads are in warp 0):
```
parts[tok=0] after reduce = 0.5 + 0.0 + 0.0 + 0.0 = 0.5   (all threads hold 0.5)
parts[tok=1] after reduce = 0.5 * 0 + 0.5 * 1 + 0 + 0 = 0.5  (all threads hold 0.5)
```

---

### Step 5: Cross-Warp Reduce → Final QK Scores

With `HEAD_DIM=128` (the real kernel), there are 4 warps. Each warp's reduce only summed its 32 dimensions. The 4 partial sums must be combined.

Lane 0 of each warp writes its partial sum to shared memory:
```cpp
if (lane_id == 0) {
    #pragma unroll
    for (int tok = 0; tok < PAGE_SIZE; tok++)
        sm.warp[warp_id * PAGE_SIZE + tok] = parts[tok];
}
__syncthreads();  // ← sync 2: all warp partial sums visible
```

Then the first `PAGE_SIZE` threads (which all fit in warp 0) sum across warps:
```cpp
if (tid < PAGE_SIZE) {
    float score = 0.0f;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; w++)
        score += sm.warp[w * PAGE_SIZE + tid];
    sm.qk[tid] = (tid < valid) ? score : -1e9f;
}
__syncthreads();  // ← sync 3: final QK scores visible to all threads
```

In our example (1 warp, so `sm.warp` holds just one warp's sums):
```
sm.qk[0] = 0.5   (dot product q · K[0])
sm.qk[1] = 0.5   (dot product q · K[1])
```

---

### Step 6: Softmax Stats (Registers Only)

All threads independently read `sm.qk` and compute the same scalar stats. No sync needed because this is a read-only step — and redundant computation across threads is cheaper than extra shared memory writes and barriers.

```cpp
float m_block = -1e9f;
for (int tok = 0; tok < PAGE_SIZE; tok++)
    m_block = fmaxf(m_block, sm.qk[tok]);   // local max for numerical stability

float exp_weights[PAGE_SIZE];
float l_block = 0.0f;
for (int tok = 0; tok < PAGE_SIZE; tok++) {
    exp_weights[tok] = expf(sm.qk[tok] - m_block);
    l_block += exp_weights[tok];
}
```

Tracing our example for Page 0:
```
m_block = max(0.5, 0.5) = 0.5
exp_weights[0] = exp(0.5 - 0.5) = exp(0) = 1.0
exp_weights[1] = exp(0.5 - 0.5) = exp(0) = 1.0
l_block = 2.0
```

---

### Step 7: Load V Into Shared Memory

The K tile in `sm.kv` is no longer needed. The kernel reuses the same memory for the V tile (the two regions `sm.warp` and `sm.qk` are separate, so the scores computed in Step 5 survive):

```cpp
// Load V page into sm.kv (overwriting K tile — sm.qk is unaffected)
for (int tok = 0; tok < PAGE_SIZE; tok++) {
    sm.kv[tok * HEAD_DIM + tid] = __half2float(v_base[tok * stride_kvs + tid]);
}
__syncthreads();  // ← sync 4: V tile visible before accumulation
```

After loading Page 0 in our example:
```
smem_kv: [0.5, 0.5, 0.5, 0.5,   0.5, 0.5, 0.5, 0.5]
          V[0][0..3]               V[1][0..3]
```

---

### Step 8: Accumulate Weighted V

Each thread accumulates the output for its own dimension:

```cpp
float acc_block = 0.0f;
#pragma unroll
for (int tok = 0; tok < PAGE_SIZE; tok++)
    acc_block += exp_weights[tok] * sm.kv[tok * HEAD_DIM + tid];
```

Thread 0 (dim d=0):
```
acc_block = 1.0 * V[0][0] + 1.0 * V[1][0]
          = 1.0 * 0.5     + 1.0 * 0.5
          = 1.0
```
Same for threads 1, 2, 3.

---

### Step 9: Online Softmax Merge (Across Pages)

After processing each page, the kernel merges the new page's stats with the running totals. This is the **online softmax** trick — it lets us compute softmax incrementally without storing all scores at once:

```cpp
const float m_new = fmaxf(m_i, m_block);
const float alpha  = expf(m_i     - m_new);   // rescale old running sum
const float beta   = expf(m_block - m_new);   // rescale new page sum
l_i = alpha * l_i + beta * l_block;
acc = alpha * acc + beta * acc_block;
m_i = m_new;
```

After Page 0 (starting from `m_i = -inf`, `l_i = 0`, `acc = 0`):
```
m_new = max(-inf, 0.5) = 0.5
alpha = exp(-inf - 0.5) ≈ 0.0
beta  = exp(0.5 - 0.5)  = 1.0
l_i   = 0*0 + 1.0*2.0  = 2.0
acc   = 0*0 + 1.0*1.0  = 1.0
m_i   = 0.5
```

Pages 1 produces the same scores and weights, so after merging Page 1:
```
m_new = max(0.5, 0.5) = 0.5
alpha = 1.0, beta = 1.0
l_i   = 1.0*2.0 + 1.0*2.0 = 4.0
acc   = 1.0*1.0 + 1.0*1.0 = 2.0
```

---

### Step 10: Write Output

After all pages, each thread normalizes and writes its dimension:

```cpp
O[batch_idx * stride_ob + q_head_idx * stride_oh + tid] = __float2half(acc / l_i);
```

Thread 0 writes: `2.0 / 4.0 = 0.5` → stored as fp16.
All threads write `0.5`. Output = `[0.5, 0.5, 0.5, 0.5]`. ✓

---

### Naive Kernel: Full Sync Accounting

Per page: **4 `__syncthreads()` calls** (K load → QK reduce → score finalize → V load).

With 50 pages: 200 syncs total. Each sync stalls all threads until the slowest thread catches up. For long sequences this is the primary cost.

---

## The Optimized Kernel: Split-KV

The naive kernel assigns all pages to one block. Long sequences (many pages) leave other SMs idle while one SM does all the work.

The split-KV approach uses two kernels:

```
Kernel 1: decode_attn_partition_kernel
  Grid: (batch, q_heads, split_kv)   ← new third dimension
  Each block handles only [start_page, end_page) — a slice of pages
  Output: fp32 scratch buffers (partial_O, partial_m, partial_l)

Kernel 2: decode_attn_reduce_kernel
  Grid: (batch, q_heads)
  Each block merges split_kv partial results into the final output
```

### Partition Kernel (Kernel 1)

Identical to the naive kernel except:
1. `blockIdx.z` is the split index — each block reads only its assigned pages:
   ```cpp
   const int split_idx        = blockIdx.z;
   const int pages_per_split  = (num_pages + split_kv - 1) / split_kv;
   const int split_start      = split_idx * pages_per_split;
   const int split_end        = min(split_start + pages_per_split, num_pages);
   ```
2. Output goes to scratch buffers instead of the final tensor:
   ```cpp
   partial_O[po_base + tid] = acc;    // fp32, not fp16
   partial_m[...] = m_i;
   partial_l[...] = l_i;
   ```

In our example with `split_kv=2`: block 0 handles Page 0, block 1 handles Page 1 — running simultaneously on two SMs.

After the partition kernel:
```
split 0: partial_m=0.5, partial_l=2.0, partial_O=[1.0, 1.0, 1.0, 1.0]
split 1: partial_m=0.5, partial_l=2.0, partial_O=[1.0, 1.0, 1.0, 1.0]
```

### Reduce Kernel (Kernel 2)

One block per (batch, head). Each thread independently merges the `split_kv` partial outputs for its own dimension. No shared memory needed — all values fit in registers:

```cpp
// Find the global max across all splits
float m_global = -1e9f;
for (int s = 0; s < split_kv; s++)
    m_global = fmaxf(m_global, partial_m[... + s]);

// Weighted sum of partial outputs
float l_global = 0.0f, acc = 0.0f;
for (int s = 0; s < split_kv; s++) {
    float w = expf(partial_m[s] - m_global);
    l_global += w * partial_l[s];
    acc      += w * partial_O[... + tid];
}

O[...] = __float2half(acc / l_global);
```

All threads compute the same `m_global` and `l_global` scalars redundantly (each thread reads the same `partial_m`/`partial_l` from HBM), but this avoids shared memory and syncs for what is a tiny loop over `split_kv ≤ 16` iterations.

Tracing our example:
```
m_global = max(0.5, 0.5) = 0.5
w0 = exp(0.5 - 0.5) = 1.0, w1 = 1.0
l_global = 1.0*2.0 + 1.0*2.0 = 4.0
acc (dim 0) = 1.0*1.0 + 1.0*1.0 = 2.0
output[0] = 2.0 / 4.0 = 0.5  ✓
```

---

## Summary: Data Flow at a Glance

```
HBM (Q tensor)
  │
  │ each thread reads one fp16 element → converts to fp32
  ▼
Registers: q_val  ←── stays here forever
  
  ┌─── Page loop ─────────────────────────────────────────────────────────┐
  │                                                                        │
  │  HBM (K page)  ──fp16→fp32──►  smem_kv  ──sync──►  Registers: parts  │
  │                                                                        │
  │  Registers: parts  ──shfl_xor_sync──►  warp sum                       │
  │  warp sum  ──lane0──►  smem_warp  ──sync──►  Registers: score         │
  │  score  ──►  smem_qk  ──sync  (all threads read same qk values)       │
  │                                                                        │
  │  Registers: exp_weights, acc_block  (softmax stats, no sync)          │
  │                                                                        │
  │  HBM (V page)  ──fp16→fp32──►  smem_kv  ──sync──►  Registers: acc    │
  │                                                                        │
  │  Registers: m_i, l_i, acc  updated via online softmax merge           │
  └────────────────────────────────────────────────────────────────────────┘
  
Registers: acc / l_i  ──fp32→fp16──►  HBM (O tensor)
```

The naive kernel does this in **one block per sequence**. The split-KV kernel parallelizes the page loop across multiple blocks, then merges with a second lightweight kernel.

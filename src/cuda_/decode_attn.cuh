/**
 * decode_attn.cuh — device-side utilities for the decode attention CUDA kernels.
 *
 * Contains:
 *   - Warp-level reductions (sum and max) using __shfl_xor_sync
 *   - Shared-memory layout constants
 *   - Inline device helpers
 *
 * All code here compiles only on the device side (nvcc / __device__ functions).
 * Host-side launchers are in attention_ext.cu.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// ── Warp-level reductions ─────────────────────────────────────────────────
//
// __shfl_xor_sync(mask, val, delta) exchanges registers between lanes whose
// IDs differ by delta.  A butterfly XOR pattern with delta = 16, 8, 4, 2, 1
// produces a full warp reduction in 5 steps — no shared memory, no sync.

__device__ __forceinline__
float warp_reduce_sum(float val) {
    // After this, ALL lanes in the warp hold the sum (broadcast result).
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, delta);
    return val;
}

__device__ __forceinline__
float warp_reduce_max(float val) {
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, delta));
    return val;
}

// ── Shared memory layout helper ───────────────────────────────────────────
//
// For a block of HEAD_DIM threads processing a KV page of PAGE_SIZE tokens:
//
//   Region 0 : smem_kv   [PAGE_SIZE * HEAD_DIM] floats — KV tile (K or V, reused)
//   Region 1 : smem_warp [NUM_WARPS * PAGE_SIZE] floats — inter-warp partial QK sums
//   Region 2 : smem_qk   [PAGE_SIZE]             floats — final QK scores
//
// Total shared memory per block:
//   (PAGE_SIZE * HEAD_DIM + NUM_WARPS * PAGE_SIZE + PAGE_SIZE) * 4 bytes
// For PAGE_SIZE=16, HEAD_DIM=128, NUM_WARPS=4: (2048 + 64 + 16)*4 = 8 512 bytes.

template<int HEAD_DIM, int PAGE_SIZE>
struct SmemLayout {
    static constexpr int NUM_WARPS   = HEAD_DIM / 32;
    static constexpr int KV_FLOATS   = PAGE_SIZE * HEAD_DIM;
    static constexpr int WARP_FLOATS = NUM_WARPS * PAGE_SIZE;
    static constexpr int QK_FLOATS   = PAGE_SIZE;
    static constexpr int TOTAL_FLOATS = KV_FLOATS + WARP_FLOATS + QK_FLOATS;
    static constexpr int BYTES        = TOTAL_FLOATS * sizeof(float);

    float* kv;    // base pointer into shared memory
    float* warp;  // kv + KV_FLOATS
    float* qk;    // warp + WARP_FLOATS

    __device__ __forceinline__
    SmemLayout(float* smem_base)
        : kv   (smem_base),
          warp (smem_base + KV_FLOATS),
          qk   (smem_base + KV_FLOATS + WARP_FLOATS)
    {}
};

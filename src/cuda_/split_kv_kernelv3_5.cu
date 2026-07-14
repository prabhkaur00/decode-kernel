/**
 * split_kv_kernelv3_5.cu — CUDA decode attention, split-KV two-pass variant
 * combining v3's software-pipelined K/V loads (cp.async / LDGSTS) with
 * v2's KV-head-centric group fusion (one K/V page load serves all
 * GROUP_SIZE query heads in the group).
 *
 * Grid: (batch, num_kv_heads, split_kv) — one CTA per kv_head, not per
 * q_head. GROUP_SIZE is a compile-time template param so the per-g loops
 * over q_vals[]/m_i[]/l_i[]/acc[] can be #pragma unroll'd.
 *
 * Pipeline: NUM_STAGES = 2 (double buffered). Each stage holds one full page
 * of both K and V, shared across all GROUP_SIZE q_heads, laid out as fp16
 * tiles in shared memory. LDGSTS uses the 16-byte variant (L1 BYPASS mode)
 * to avoid polluting L1 with streaming KV.
 *
 * Prologue fills the pipeline with the first NUM_STAGES pages, then the main
 * loop waits on the current stage, fuses per-g score/softmax/accumulate
 * (same structure as v2.5 — QK scores for the in-flight g live in a small
 * shared scratch, not a per-thread qk_scores[GROUP_SIZE][PAGE_SIZE] array),
 * and prefetches a future page into the just-consumed stage buffer.
 *
 * Host caller MUST set the larger shared-memory carveout via
 * cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
 *                      required_smem_bytes) before launch, otherwise the
 * default 48 KB limit will reduce occupancy.
 */

#include "decode_attn.cuh"
#include <torch/extension.h>
#include <nvToolsExt.h>
#include <cuda_pipeline.h>

// ── Pass 1: partition kernel (pipelined) ──────────────────────────────────
template<int HEAD_DIM, int PAGE_SIZE, int GROUP_SIZE>
__global__ void decode_attn_partition_v4_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ KV,
    const int*    __restrict__ kv_indptr,
    const int*    __restrict__ kv_indices,
    const int*    __restrict__ kv_last_len,
    float* __restrict__ partial_O,   // (B, H_q, SPLIT_KV, D)
    float* __restrict__ partial_m,   // (B, H_q, SPLIT_KV)
    float* __restrict__ partial_l,   // (B, H_q, SPLIT_KV)
    int stride_qb,  int stride_qh,
    int stride_kvp, int stride_kvr, int stride_kvs, int stride_kvh,
    int stride_pob, int stride_poh, int stride_pos,
    int stride_pmb, int stride_pmh,
    float scale,
    int split_kv
) {
    constexpr int NUM_STAGES   = 2;
    constexpr int NUM_WARPS    = HEAD_DIM / 32;
    constexpr int TILE_ELEMS   = PAGE_SIZE * HEAD_DIM;   // per K or V tile, in halfs
    constexpr int VEC          = 8;                       // 8 halfs = 16 bytes per cp.async
    constexpr int VECS_PER_TOK = HEAD_DIM / VEC;
    constexpr int TOTAL_VECS   = PAGE_SIZE * VECS_PER_TOK;

    static_assert(HEAD_DIM % VEC == 0, "HEAD_DIM must be multiple of 8 for 16-byte cp.async");

    const int batch_idx   = blockIdx.x;
    const int kv_head_idx = blockIdx.y;   // group-fused: one CTA per kv_head, not q_head
    const int split_idx   = blockIdx.z;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    // ── Shared memory layout ────────────────────────────────────────────
    // [ stage0_K | stage0_V | stage1_K | stage1_V | qk | warp_scratch ]
    // One K/V tile pair per stage, SHARED across all GROUP_SIZE q_heads —
    // this is the group-fusion win (load once, not once per q_head) laid
    // on top of the pipeline's double buffering (load next while using
    // current). sm_qk/sm_warp are the same PAGE_SIZE-sized scratch reused
    // per-g inside the fused loop, exactly as in v3 — no per-g stash.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* stage_k[NUM_STAGES];
    __half* stage_v[NUM_STAGES];
    stage_k[0] = reinterpret_cast<__half*>(smem_raw);
    stage_v[0] = stage_k[0] + TILE_ELEMS;
    stage_k[1] = stage_v[0] + TILE_ELEMS;
    stage_v[1] = stage_k[1] + TILE_ELEMS;
    float* sm_qk   = reinterpret_cast<float*>(stage_v[1] + TILE_ELEMS);
    float* sm_warp = sm_qk + PAGE_SIZE;

    // ── Load all GROUP_SIZE Q vectors once ──────────────────────────────
    float q_vals[GROUP_SIZE];
    #pragma unroll
    for (int g = 0; g < GROUP_SIZE; g++) {
        const int q_head_idx = kv_head_idx * GROUP_SIZE + g;
        q_vals[g] = __half2float(Q[batch_idx * stride_qb + q_head_idx * stride_qh + tid]) * scale;
    }

    // ── Sequence bookkeeping ────────────────────────────────────────────
    const int seq_start = kv_indptr[batch_idx];
    const int seq_end   = kv_indptr[batch_idx + 1];
    const int last_len  = kv_last_len[batch_idx];
    const int num_pages = seq_end - seq_start;

    const int pages_per_split = (num_pages + split_kv - 1) / split_kv;
    const int split_start     = split_idx * pages_per_split;
    const int split_end       = min(split_start + pages_per_split, num_pages);
    const int my_num_pages    = split_end - split_start;

    // Early exit for empty partitions — write for every g in the group.
    if (my_num_pages <= 0) {
        #pragma unroll
        for (int g = 0; g < GROUP_SIZE; g++) {
            const int q_head_idx = kv_head_idx * GROUP_SIZE + g;
            const int po_base =
                batch_idx * stride_pob + q_head_idx * stride_poh + split_idx * stride_pos;
            partial_O[po_base + tid] = 0.0f;
            if (tid == 0) {
                partial_m[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = -1e9f;
                partial_l[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = 0.0f;
            }
        }
        return;
    }

    // ── Helpers (kv_head-indexed — shared across the whole group) ──────
    auto k_base_of = [&](int gp) {
        int page = kv_indices[seq_start + split_start + gp];
        return KV + page * stride_kvp + 0 * stride_kvr + kv_head_idx * stride_kvh;
    };
    auto v_base_of = [&](int gp) {
        int page = kv_indices[seq_start + split_start + gp];
        return KV + page * stride_kvp + 1 * stride_kvr + kv_head_idx * stride_kvh;
    };
    auto valid_of = [&](int gp) {
        return (split_start + gp == num_pages - 1) ? last_len : PAGE_SIZE;
    };

    auto async_load_tile = [&](__half* dst, const __half* base, int valid) {
        #pragma unroll
        for (int i = tid; i < TOTAL_VECS; i += HEAD_DIM) {
            int tok = i / VECS_PER_TOK;
            int d   = (i % VECS_PER_TOK) * VEC;
            if (tok < valid) {
                __pipeline_memcpy_async(
                    dst  + tok * HEAD_DIM   + d,
                    base + tok * stride_kvs + d,
                    16
                );
            }
        }
    };

    // ── Prologue: fill the pipeline (one K+V pair per stage, group-shared) ─
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        if (s < my_num_pages) {
            async_load_tile(stage_k[s], k_base_of(s), valid_of(s));
            async_load_tile(stage_v[s], v_base_of(s), valid_of(s));
        }
        __pipeline_commit();
    }

    // ── Running softmax state — one set per q_head in the group ────────
    float m_i[GROUP_SIZE];
    float l_i[GROUP_SIZE];
    float acc[GROUP_SIZE];
    #pragma unroll
    for (int g = 0; g < GROUP_SIZE; g++) {
        m_i[g] = -1e9f;
        l_i[g] = 0.0f;
        acc[g] = 0.0f;
    }

    int stage = 0;

    // ── Main loop ───────────────────────────────────────────────────────
    for (int p = 0, fetch = NUM_STAGES; p < my_num_pages; p++, fetch++) {
        __pipeline_wait_prior(NUM_STAGES - 1);
        __syncthreads();

        const int this_valid = valid_of(p);
        __half* k_tile = stage_k[stage];
        __half* v_tile = stage_v[stage];   // already resident — this is what
                                            // lets the fused per-g loop below
                                            // consume V right after scoring,
                                            // same as v3's sm.kv_v, but now
                                            // arrived via cp.async instead of
                                            // a blocking synchronous load.

        // ── Fused per-g: score compute -> softmax -> accumulate ─────────
        // Same structure as v3: only PAGE_SIZE (16-float) qk buffer live
        // at a time, reused across g, instead of qk_scores[GROUP_SIZE][PAGE_SIZE].
        #pragma unroll
        for (int g = 0; g < GROUP_SIZE; g++) {
            float parts[PAGE_SIZE];
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; tok++) {
                float k_val = __half2float(k_tile[tok * HEAD_DIM + tid]);
                parts[tok] = warp_reduce_sum(q_vals[g] * k_val);
            }

            if (lane_id == 0) {
                #pragma unroll
                for (int tok = 0; tok < PAGE_SIZE; tok++)
                    sm_warp[warp_id * PAGE_SIZE + tok] = parts[tok];
            }
            __syncthreads();

            if (tid < PAGE_SIZE) {
                float score = 0.0f;
                #pragma unroll
                for (int w = 0; w < NUM_WARPS; w++)
                    score += sm_warp[w * PAGE_SIZE + tid];
                sm_qk[tid] = (tid < this_valid) ? score : -1e9f;
            }
            __syncthreads();

            float m_block = -1e9f;
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; tok++)
                m_block = fmaxf(m_block, sm_qk[tok]);

            float exp_weights[PAGE_SIZE];
            float l_block = 0.0f;
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; tok++) {
                exp_weights[tok] = expf(sm_qk[tok] - m_block);
                l_block += exp_weights[tok];
            }

            float acc_block = 0.0f;
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; tok++) {
                float v_val = __half2float(v_tile[tok * HEAD_DIM + tid]);
                acc_block += exp_weights[tok] * v_val;
            }

            const float m_new = fmaxf(m_i[g], m_block);
            const float alpha = expf(m_i[g]  - m_new);
            const float beta  = expf(m_block - m_new);
            l_i[g] = alpha * l_i[g] + beta * l_block;
            acc[g] = alpha * acc[g] + beta * acc_block;
            m_i[g] = m_new;
            // sm_warp/sm_qk overwritten next g iteration — guarded by the
            // two syncthreads() above, same as v3.
        }

        // ── Prefetch future page into just-consumed stage buffer ────────
        // Sync before overwriting: the last g iteration's reads of k_tile/
        // v_tile must complete before this stage buffer is reused for the
        // next page's cp.async destination.
        __syncthreads();
        if (fetch < my_num_pages) {
            async_load_tile(stage_k[stage], k_base_of(fetch), valid_of(fetch));
            async_load_tile(stage_v[stage], v_base_of(fetch), valid_of(fetch));
        }
        __pipeline_commit();

        stage = (stage + 1) % NUM_STAGES;
    }

    __pipeline_wait_prior(0);

    // ── Write partials for all q_heads in the group ─────────────────────
    #pragma unroll
    for (int g = 0; g < GROUP_SIZE; g++) {
        const int q_head_idx = kv_head_idx * GROUP_SIZE + g;
        const int po_base =
            batch_idx * stride_pob + q_head_idx * stride_poh + split_idx * stride_pos;
        partial_O[po_base + tid] = acc[g];
        partial_m[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = m_i[g];
        partial_l[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = l_i[g];
    }
}

// ── Pass 2: reduction kernel (unchanged) ──────────────────────────────────

template<int HEAD_DIM>
__global__ void decode_attn_reduce_kernel(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __half*      __restrict__ O,
    int stride_pob, int stride_poh, int stride_pos,
    int stride_pmb, int stride_pmh,
    int stride_ob,  int stride_oh,
    int split_kv
) {
    const int batch_idx  = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int tid        = threadIdx.x;

    float m_global = -1e9f;
    for (int s = 0; s < split_kv; s++) {
        const float m_s =
            partial_m[batch_idx * stride_pmb + q_head_idx * stride_pmh + s];
        m_global = fmaxf(m_global, m_s);
    }

    float l_global = 0.0f;
    float acc      = 0.0f;
    for (int s = 0; s < split_kv; s++) {
        const float m_s =
            partial_m[batch_idx * stride_pmb + q_head_idx * stride_pmh + s];
        const float l_s =
            partial_l[batch_idx * stride_pmb + q_head_idx * stride_pmh + s];
        const float o_s =
            partial_O[batch_idx * stride_pob + q_head_idx * stride_poh
                      + s * stride_pos + tid];
        const float w = expf(m_s - m_global);
        l_global += w * l_s;
        acc      += w * o_s;
    }

    O[batch_idx * stride_ob + q_head_idx * stride_oh + tid] =
        __float2half(acc / l_global);
}

// ── Host-side launcher (partition) ────────────────────────────────────────
//
// Grid is (batch, num_kv_heads, split_kv) — one CTA per kv_head (group-fused,
// same as v2/v2.5), not per q_head as in v3. GROUP_SIZE is a compile-time
// template param (needed for #pragma unroll over q_vals[]/m_i[]/etc.), so
// dispatch goes through DISPATCH_GROUP same as v2/v2.5, and group_size is
// no longer passed as a runtime kernel argument.

template<int HEAD_DIM, int GROUP_SIZE>
static void launch_partition_v3_5_hd(
    const __half* q_ptr, const __half* kv_ptr,
    const int* indptr, const int* indices, const int* last_len,
    float* po, float* pm, float* pl,
    int stride_qb, int stride_qh,
    int stride_kvp, int stride_kvr, int stride_kvs, int stride_kvh,
    int stride_pob, int stride_poh, int stride_pos,
    int stride_pmb, int stride_pmh,
    float scale, int split_kv,
    int batch, int num_kv_heads
) {
    constexpr int PAGE_SIZE  = 16;
    constexpr int NUM_STAGES = 2;
    constexpr int NUM_WARPS  = HEAD_DIM / 32;

    // Shared memory: 2 stages of (K + V) fp16 tiles + qk scratch + warp scratch
    // (qk/warp are reused across g inside the fused loop, not sized by
    // GROUP_SIZE — same PAGE_SIZE-sized scratch as v3).
    constexpr int smem_bytes =
        NUM_STAGES * 2 * PAGE_SIZE * HEAD_DIM * sizeof(__half)   // K + V tiles
        + PAGE_SIZE * sizeof(float)                                // sm_qk
        + NUM_WARPS * PAGE_SIZE * sizeof(float);                   // sm_warp

    auto kernel = decode_attn_partition_v4_kernel<HEAD_DIM, PAGE_SIZE, GROUP_SIZE>;

    // Opt into larger shared memory carveout on A100+ (needed to keep occupancy up).
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        attr_set = true;
    }

    kernel<<<dim3(batch, num_kv_heads, split_kv), HEAD_DIM, smem_bytes>>>(
        q_ptr, kv_ptr, indptr, indices, last_len,
        po, pm, pl,
        stride_qb, stride_qh,
        stride_kvp, stride_kvr, stride_kvs, stride_kvh,
        stride_pob, stride_poh, stride_pos,
        stride_pmb, stride_pmh,
        scale, split_kv
    );
}

// ── Dispatch macro (GROUP_SIZE known at kernel compile time) ──────────────

#define DISPATCH_GROUP(HD, GS, ...)                                  \
    if      (GS == 1) launch_partition_v3_5_hd<HD, 1>(__VA_ARGS__);  \
    else if (GS == 2) launch_partition_v3_5_hd<HD, 2>(__VA_ARGS__);  \
    else if (GS == 4) launch_partition_v3_5_hd<HD, 4>(__VA_ARGS__);  \
    else if (GS == 8) launch_partition_v3_5_hd<HD, 8>(__VA_ARGS__);  \
    else TORCH_CHECK(false, "Unsupported group_size: ", GS,          \
                     ". Must be 1, 2, 4, or 8.")

// ── Host-side launcher (reduction, unchanged) ─────────────────────────────

template<int HEAD_DIM>
static void launch_reduce_hd(
    const float* po, const float* pm, const float* pl,
    __half* o_ptr,
    int stride_pob, int stride_poh, int stride_pos,
    int stride_pmb, int stride_pmh,
    int stride_ob, int stride_oh,
    int split_kv, int batch, int num_q_heads
) {
    decode_attn_reduce_kernel<HEAD_DIM>
        <<<dim3(batch, num_q_heads), HEAD_DIM>>>(
            po, pm, pl, o_ptr,
            stride_pob, stride_poh, stride_pos,
            stride_pmb, stride_pmh,
            stride_ob, stride_oh,
            split_kv
        );
}

// ── PyTorch extension entry point ─────────────────────────────────────────

torch::Tensor decode_attention_split_kv_v3_5_cuda(
    torch::Tensor q,
    torch::Tensor kv_data,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    torch::Tensor kv_last_page_len,
    int split_kv
) {
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(kv_data.size(2) == 16, "page_size must be 16");
    TORCH_CHECK(split_kv >= 1 && split_kv <= 32, "split_kv must be in [1, 32]");

    const int batch        = q.size(0);
    const int num_q_heads  = q.size(1);
    const int head_dim     = q.size(2);
    const int num_kv_heads = kv_data.size(3);
    const int group_size   = num_q_heads / num_kv_heads;
    const float scale      = 1.0f / sqrtf(static_cast<float>(head_dim));

    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "num_q_heads must be divisible by num_kv_heads");

    // Alignment check for LDGSTS 16-byte variant: stride_kvs (in halfs)
    // must be a multiple of 8 so that byte offsets are multiples of 16.
    TORCH_CHECK(kv_data.stride(2) % 8 == 0,
                "kv_data token stride must be a multiple of 8 halfs for cp.async");

    auto out       = torch::empty_like(q);
    auto partial_O = torch::empty({batch, num_q_heads, split_kv, head_dim},
                                   torch::dtype(torch::kFloat32).device(q.device()));
    auto partial_m = torch::empty({batch, num_q_heads, split_kv},
                                   torch::dtype(torch::kFloat32).device(q.device()));
    auto partial_l = torch::empty_like(partial_m);

    auto q_ptr  = reinterpret_cast<const __half*>(q.data_ptr());
    auto kv_ptr = reinterpret_cast<const __half*>(kv_data.data_ptr());
    auto o_ptr  = reinterpret_cast<__half*>(out.data_ptr());
    auto po_ptr = partial_O.data_ptr<float>();
    auto pm_ptr = partial_m.data_ptr<float>();
    auto pl_ptr = partial_l.data_ptr<float>();

#define PART_ARGS q_ptr, kv_ptr,                               \
    kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(),     \
    kv_last_page_len.data_ptr<int>(),                          \
    po_ptr, pm_ptr, pl_ptr,                                    \
    (int)q.stride(0),           (int)q.stride(1),              \
    (int)kv_data.stride(0),     (int)kv_data.stride(1),        \
    (int)kv_data.stride(2),     (int)kv_data.stride(3),        \
    (int)partial_O.stride(0),   (int)partial_O.stride(1),      \
    (int)partial_O.stride(2),                                  \
    (int)partial_m.stride(0),   (int)partial_m.stride(1),      \
    scale, split_kv, batch, num_kv_heads

    nvtxRangePushA("cuda_split_kv_partition_pipelined_v3_5");
    if (head_dim == 64) {
        DISPATCH_GROUP(64, group_size, PART_ARGS);
    } else if (head_dim == 128) {
        DISPATCH_GROUP(128, group_size, PART_ARGS);
    } else if (head_dim == 256) {
        DISPATCH_GROUP(256, group_size, PART_ARGS);
    } else {
        TORCH_CHECK(false, "head_dim must be 64, 128, or 256; got ", head_dim);
    }
    nvtxRangePop();
#undef PART_ARGS

#define RED_ARGS po_ptr, pm_ptr, pl_ptr, o_ptr,                \
    (int)partial_O.stride(0), (int)partial_O.stride(1),        \
    (int)partial_O.stride(2),                                  \
    (int)partial_m.stride(0), (int)partial_m.stride(1),        \
    (int)out.stride(0),       (int)out.stride(1),              \
    split_kv, batch, num_q_heads

    nvtxRangePushA("cuda_split_kv_reduce");
    if      (head_dim ==  64) launch_reduce_hd< 64>(RED_ARGS);
    else if (head_dim == 128) launch_reduce_hd<128>(RED_ARGS);
    else if (head_dim == 256) launch_reduce_hd<256>(RED_ARGS);
    else TORCH_CHECK(false, "head_dim must be 64, 128, or 256; got ", head_dim);
    nvtxRangePop();
#undef RED_ARGS

    return out;
}

// ── pybind11 module ───────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA decode attention — split-KV two-pass kernel (v3.5: pipelined + group-fused)";
    m.def("decode_attention_split_kv",
          &decode_attention_split_kv_v3_5_cuda,
          "Split-KV v3.5: pipelined + group-fused partition + reduction",
          py::arg("q"), py::arg("kv_data"), py::arg("kv_indptr"),
          py::arg("kv_indices"), py::arg("kv_last_page_len"),
          py::arg("split_kv") = 4);
}

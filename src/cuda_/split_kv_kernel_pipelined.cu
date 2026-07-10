/**
 * split_kv_kernel_pipelined.cu — CUDA decode attention, split-KV two-pass
 * variant with software-pipelined K/V loads via cp.async (LDGSTS).
 *
 * Pipeline: NUM_STAGES = 2 (double buffered). Each stage holds one full page
 * of both K and V, laid out as fp16 tiles in shared memory. LDGSTS uses the
 * 16-byte variant (L1 BYPASS mode) to avoid polluting L1 with streaming KV.
 *
 * Prologue fills the pipeline with the first NUM_STAGES pages, then the main
 * loop waits on the current stage, computes QK + V-accumulate, and prefetches
 * a future page into the just-consumed stage buffer.
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

template<int HEAD_DIM, int PAGE_SIZE>
__global__ void decode_attn_partition_kernel_pipelined(
    const __half* __restrict__ Q,
    const __half* __restrict__ KV,
    const int*    __restrict__ kv_indptr,
    const int*    __restrict__ kv_indices,
    const int*    __restrict__ kv_last_len,
    float* __restrict__ partial_O,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int stride_qb,  int stride_qh,
    int stride_kvp, int stride_kvr, int stride_kvs, int stride_kvh,
    int stride_pob, int stride_poh, int stride_pos,
    int stride_pmb, int stride_pmh,
    float scale,
    int group_size,
    int split_kv
) {
    constexpr int NUM_STAGES  = 2;
    constexpr int NUM_WARPS   = HEAD_DIM / 32;
    constexpr int TILE_ELEMS  = PAGE_SIZE * HEAD_DIM;   // per K or V tile, in halfs
    constexpr int VEC         = 8;                       // 8 halfs = 16 bytes per cp.async
    constexpr int VECS_PER_TOK = HEAD_DIM / VEC;
    constexpr int TOTAL_VECS   = PAGE_SIZE * VECS_PER_TOK;

    static_assert(HEAD_DIM % VEC == 0, "HEAD_DIM must be multiple of 8 for 16-byte cp.async");

    const int batch_idx   = blockIdx.x;
    const int q_head_idx  = blockIdx.y;
    const int split_idx   = blockIdx.z;
    const int kv_head_idx = q_head_idx / group_size;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    // ── Shared memory layout ────────────────────────────────────────────
    // [ stage0_K | stage0_V | stage1_K | stage1_V | qk | warp_scratch ]
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* stage_k[NUM_STAGES];
    __half* stage_v[NUM_STAGES];
    stage_k[0] = reinterpret_cast<__half*>(smem_raw);
    stage_v[0] = stage_k[0] + TILE_ELEMS;
    stage_k[1] = stage_v[0] + TILE_ELEMS;
    stage_v[1] = stage_k[1] + TILE_ELEMS;
    float* sm_qk   = reinterpret_cast<float*>(stage_v[1] + TILE_ELEMS);
    float* sm_warp = sm_qk + PAGE_SIZE;

    // ── Load Q once ─────────────────────────────────────────────────────
    const float q_val =
        __half2float(Q[batch_idx * stride_qb + q_head_idx * stride_qh + tid]) * scale;

    // ── Sequence bookkeeping ────────────────────────────────────────────
    const int seq_start = kv_indptr[batch_idx];
    const int seq_end   = kv_indptr[batch_idx + 1];
    const int last_len  = kv_last_len[batch_idx];
    const int num_pages = seq_end - seq_start;

    const int pages_per_split = (num_pages + split_kv - 1) / split_kv;
    const int split_start     = split_idx * pages_per_split;
    const int split_end       = min(split_start + pages_per_split, num_pages);
    const int my_num_pages    = split_end - split_start;

    // Early exit for empty partitions
    if (my_num_pages <= 0) {
        const int po_base =
            batch_idx * stride_pob + q_head_idx * stride_poh + split_idx * stride_pos;
        partial_O[po_base + tid] = 0.0f;
        if (tid == 0) {
            partial_m[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = -1e9f;
            partial_l[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = 0.0f;
        }
        return;
    }

    // ── Helpers ─────────────────────────────────────────────────────────
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

    // Issue LDGSTS for one tile (K or V of one page) into a stage buffer.
    // Each thread issues its share of 16-byte cp.async instructions.
    // Out-of-range tokens are NOT zero-filled here; QK masking handles them.
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

    // ── Prologue: fill the pipeline ─────────────────────────────────────
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        if (s < my_num_pages) {
            async_load_tile(stage_k[s], k_base_of(s), valid_of(s));
            async_load_tile(stage_v[s], v_base_of(s), valid_of(s));
        }
        __pipeline_commit();   // ALWAYS commit, even if nothing was issued
    }

    // ── Running softmax state ───────────────────────────────────────────
    float m_i = -1e9f;
    float l_i = 0.0f;
    float acc = 0.0f;

    int stage = 0;

    // ── Main loop ───────────────────────────────────────────────────────
    for (int p = 0, fetch = NUM_STAGES; p < my_num_pages; p++, fetch++) {
        // Wait until only NUM_STAGES-1 commits are still in flight,
        // i.e. this page's data has landed in shared memory.
        __pipeline_wait_prior<NUM_STAGES - 1>();
        __syncthreads();

        const int this_valid = valid_of(p);

        // ── QK scores ───────────────────────────────────────────────────
        float parts[PAGE_SIZE];
        #pragma unroll
        for (int tok = 0; tok < PAGE_SIZE; tok++) {
            float k_val = __half2float(stage_k[stage][tok * HEAD_DIM + tid]);
            parts[tok] = warp_reduce_sum(q_val * k_val);
        }

        // Warp partials → cross-warp reduction via shared memory
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
            // Mask invalid tokens with -inf so exp() zeros them out
            sm_qk[tid] = (tid < this_valid) ? score : -1e9f;
        }
        __syncthreads();

        // ── Softmax stats for this page ─────────────────────────────────
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

        // ── V accumulate ────────────────────────────────────────────────
        // stage_v[stage] is already in shared memory (same stage as K).
        // Out-of-range tokens have exp_weight=0 (from -inf score), so garbage
        // V values for those tokens contribute nothing.
        float acc_block = 0.0f;
        #pragma unroll
        for (int tok = 0; tok < PAGE_SIZE; tok++) {
            float v_val = __half2float(stage_v[stage][tok * HEAD_DIM + tid]);
            acc_block += exp_weights[tok] * v_val;
        }

        // ── Online softmax merge ────────────────────────────────────────
        const float m_new = fmaxf(m_i, m_block);
        const float alpha = expf(m_i     - m_new);
        const float beta  = expf(m_block - m_new);
        l_i = alpha * l_i + beta * l_block;
        acc = alpha * acc + beta * acc_block;
        m_i = m_new;

        // ── Prefetch future page into just-consumed stage buffer ────────
        // Sync before overwriting so no thread still reads the old tile.
        __syncthreads();
        if (fetch < my_num_pages) {
            async_load_tile(stage_k[stage], k_base_of(fetch), valid_of(fetch));
            async_load_tile(stage_v[stage], v_base_of(fetch), valid_of(fetch));
        }
        __pipeline_commit();   // ALWAYS commit, keeps wait_prior<N-1> valid

        stage = (stage + 1) % NUM_STAGES;
    }

    // Drain any remaining commits so no in-flight LDGSTS outlives the kernel
    __pipeline_wait_prior<0>();

    // ── Write partials ──────────────────────────────────────────────────
    const int po_base =
        batch_idx * stride_pob + q_head_idx * stride_poh + split_idx * stride_pos;
    partial_O[po_base + tid] = acc;
    if (tid == 0) {
        partial_m[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = m_i;
        partial_l[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = l_i;
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

template<int HEAD_DIM>
static void launch_partition_hd(
    const __half* q_ptr, const __half* kv_ptr,
    const int* indptr, const int* indices, const int* last_len,
    float* po, float* pm, float* pl,
    int stride_qb, int stride_qh,
    int stride_kvp, int stride_kvr, int stride_kvs, int stride_kvh,
    int stride_pob, int stride_poh, int stride_pos,
    int stride_pmb, int stride_pmh,
    float scale, int group_size, int split_kv,
    int batch, int num_q_heads
) {
    constexpr int PAGE_SIZE  = 16;
    constexpr int NUM_STAGES = 2;
    constexpr int NUM_WARPS  = HEAD_DIM / 32;

    // Shared memory: 2 stages of (K + V) fp16 tiles + qk scratch + warp scratch
    constexpr int smem_bytes =
        NUM_STAGES * 2 * PAGE_SIZE * HEAD_DIM * sizeof(__half)   // K + V tiles
        + PAGE_SIZE * sizeof(float)                                // sm_qk
        + NUM_WARPS * PAGE_SIZE * sizeof(float);                   // sm_warp

    auto kernel = decode_attn_partition_kernel_pipelined<HEAD_DIM, PAGE_SIZE>;

    // Opt into larger shared memory carveout on A100+ (needed to keep occupancy up).
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        attr_set = true;
    }

    kernel<<<dim3(batch, num_q_heads, split_kv), HEAD_DIM, smem_bytes>>>(
        q_ptr, kv_ptr, indptr, indices, last_len,
        po, pm, pl,
        stride_qb, stride_qh,
        stride_kvp, stride_kvr, stride_kvs, stride_kvh,
        stride_pob, stride_poh, stride_pos,
        stride_pmb, stride_pmh,
        scale, group_size, split_kv
    );
}

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

torch::Tensor decode_attention_split_kv_cuda_pp(
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
    scale, group_size, split_kv, batch, num_q_heads

    nvtxRangePushA("cuda_split_kv_partition_pipelined");
    if      (head_dim ==  64) launch_partition_hd< 64>(PART_ARGS);
    else if (head_dim == 128) launch_partition_hd<128>(PART_ARGS);
    else if (head_dim == 256) launch_partition_hd<256>(PART_ARGS);
    else TORCH_CHECK(false, "head_dim must be 64, 128, or 256; got ", head_dim);
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
    m.doc() = "CUDA decode attention — split-KV two-pass kernel (pipelined)";
    m.def("decode_attention_split_kv",
          &decode_attention_split_kv_cuda_pp,
          "Split-KV decode attention: pipelined partition + reduction",
          py::arg("q"), py::arg("kv_data"), py::arg("kv_indptr"),
          py::arg("kv_indices"), py::arg("kv_last_page_len"),
          py::arg("split_kv") = 4);
}

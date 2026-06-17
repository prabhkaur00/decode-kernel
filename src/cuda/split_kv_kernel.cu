/**
 * split_kv_kernel.cu — CUDA decode attention, split-KV two-pass variant.
 *
 * Pass 1 — Partition kernel
 *   Grid : (batch, num_q_heads, split_kv)
 *   Block: HEAD_DIM threads.
 *   Each CTA scans a contiguous slice of KV pages and writes partial
 *   (m_i, l_i, O_i) to fp32 scratch buffers.
 *
 * Pass 2 — Reduction kernel
 *   Grid : (batch, num_q_heads)
 *   Block: HEAD_DIM threads.
 *   Merges split_kv partial results using the online softmax identity.
 *
 * NVTX3 ranges are placed around each pass launch so nsys shows them as
 * separate labelled bands on the CPU timeline in addition to the distinct
 * CUDA kernel names visible on the GPU row.
 */

#include "decode_attn.cuh"
#include <torch/extension.h>
#include <nvToolsExt.h>

// ── Pass 1: partition kernel ───────────────────────────────────────────────

template<int HEAD_DIM, int PAGE_SIZE>
__global__ void decode_attn_partition_kernel(
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
    int group_size,
    int split_kv
) {
    constexpr int NUM_WARPS = HEAD_DIM / 32;

    const int batch_idx   = blockIdx.x;
    const int q_head_idx  = blockIdx.y;
    const int split_idx   = blockIdx.z;
    const int kv_head_idx = q_head_idx / group_size;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    extern __shared__ float smem[];
    SmemLayout<HEAD_DIM, PAGE_SIZE> sm(smem);

    const float q_val =
        __half2float(Q[batch_idx * stride_qb + q_head_idx * stride_qh + tid]) * scale;

    const int seq_start = kv_indptr[batch_idx];
    const int seq_end   = kv_indptr[batch_idx + 1];
    const int last_len  = kv_last_len[batch_idx];
    const int num_pages = seq_end - seq_start;

    const int pages_per_split = (num_pages + split_kv - 1) / split_kv;
    const int split_start     = split_idx * pages_per_split;
    const int split_end       = min(split_start + pages_per_split, num_pages);
    const int my_num_pages    = split_end - split_start;

    float m_i = -1e9f;
    float l_i = 0.0f;
    float acc = 0.0f;

    for (int p = 0; p < my_num_pages; p++) {
        const int global_offset = split_start + p;
        const int page  = kv_indices[seq_start + global_offset];
        const bool last = (global_offset == num_pages - 1);
        const int valid = last ? last_len : PAGE_SIZE;

        // Load K
        {
            const __half* k_base =
                KV + page * stride_kvp + 0 * stride_kvr + kv_head_idx * stride_kvh;
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; tok++)
                sm.kv[tok * HEAD_DIM + tid] = (tok < valid)
                    ? __half2float(k_base[tok * stride_kvs + tid]) : 0.0f;
        }
        __syncthreads();

        // QK scores
        float parts[PAGE_SIZE];
        #pragma unroll
        for (int tok = 0; tok < PAGE_SIZE; tok++)
            parts[tok] = warp_reduce_sum(q_val * sm.kv[tok * HEAD_DIM + tid]);

        if (lane_id == 0) {
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; tok++)
                sm.warp[warp_id * PAGE_SIZE + tok] = parts[tok];
        }
        __syncthreads();

        if (tid < PAGE_SIZE) {
            float score = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++)
                score += sm.warp[w * PAGE_SIZE + tid];
            sm.qk[tid] = (tid < valid) ? score : -1e9f;
        }
        __syncthreads();

        // Softmax stats
        float m_block = -1e9f;
        #pragma unroll
        for (int tok = 0; tok < PAGE_SIZE; tok++)
            m_block = fmaxf(m_block, sm.qk[tok]);

        float exp_weights[PAGE_SIZE];
        float l_block = 0.0f;
        #pragma unroll
        for (int tok = 0; tok < PAGE_SIZE; tok++) {
            exp_weights[tok] = expf(sm.qk[tok] - m_block);
            l_block += exp_weights[tok];
        }

        // Load V
        {
            const __half* v_base =
                KV + page * stride_kvp + 1 * stride_kvr + kv_head_idx * stride_kvh;
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; tok++)
                sm.kv[tok * HEAD_DIM + tid] = (tok < valid)
                    ? __half2float(v_base[tok * stride_kvs + tid]) : 0.0f;
        }
        __syncthreads();

        // Accumulate
        float acc_block = 0.0f;
        #pragma unroll
        for (int tok = 0; tok < PAGE_SIZE; tok++)
            acc_block += exp_weights[tok] * sm.kv[tok * HEAD_DIM + tid];

        // Merge
        const float m_new = fmaxf(m_i, m_block);
        const float alpha  = expf(m_i     - m_new);
        const float beta   = expf(m_block - m_new);
        l_i = alpha * l_i + beta * l_block;
        acc = alpha * acc + beta * acc_block;
        m_i = m_new;
    }

    // Write partial results; empty partitions write m=-inf, l=0, O=0
    const int po_base =
        batch_idx * stride_pob + q_head_idx * stride_poh + split_idx * stride_pos;
    partial_O[po_base + tid] = acc;
    partial_m[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = m_i;
    partial_l[batch_idx * stride_pmb + q_head_idx * stride_pmh + split_idx] = l_i;
}

// ── Pass 2: reduction kernel ───────────────────────────────────────────────

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

// ── Host-side launchers ───────────────────────────────────────────────────

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
    constexpr int PAGE_SIZE = 16;
    const int smem = SmemLayout<HEAD_DIM, PAGE_SIZE>::BYTES;
    decode_attn_partition_kernel<HEAD_DIM, PAGE_SIZE>
        <<<dim3(batch, num_q_heads, split_kv), HEAD_DIM, smem>>>(
            q_ptr, kv_ptr, indptr, indices, last_len,
            po, pm, pl,
            stride_qb, stride_qh,
            stride_kvp, stride_kvr, stride_kvs, stride_kvh,
            stride_pob, stride_poh, stride_pos,
            stride_pmb, stride_pmh,
            scale, group_size, split_kv
        );
}

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

torch::Tensor decode_attention_split_kv_cuda(
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

    auto out      = torch::empty_like(q);
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

    // ── Pass 1: partition ─────────────────────────────────────────────
    nvtxRangePushA("cuda_split_kv_partition");
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

    // ── Pass 2: reduction ─────────────────────────────────────────────
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
    m.doc() = "CUDA decode attention — split-KV two-pass kernel";
    m.def("decode_attention_split_kv",
          &decode_attention_split_kv_cuda,
          "Split-KV decode attention: partition pass + reduction pass",
          py::arg("q"), py::arg("kv_data"), py::arg("kv_indptr"),
          py::arg("kv_indices"), py::arg("kv_last_page_len"),
          py::arg("split_kv") = 4);
}

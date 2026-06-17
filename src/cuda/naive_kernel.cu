/**
 * naive_kernel.cu — CUDA decode attention, naive single-block variant.
 *
 * Grid : (batch, num_q_heads)
 * Block: HEAD_DIM threads; thread tid owns head dimension tid.
 *
 * Each CTA reads the full KV sequence for its (batch, q_head) pair,
 * accumulating with online softmax. No split across the KV dimension.
 */

#include "decode_attn.cuh"
#include <torch/extension.h>

// ── Kernel ────────────────────────────────────────────────────────────────

template<int HEAD_DIM, int PAGE_SIZE>
__global__ void decode_attn_naive_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ KV,
    const int*    __restrict__ kv_indptr,
    const int*    __restrict__ kv_indices,
    const int*    __restrict__ kv_last_len,
    __half*       __restrict__ O,
    int stride_qb,  int stride_qh,
    int stride_kvp, int stride_kvr, int stride_kvs, int stride_kvh,
    int stride_ob,  int stride_oh,
    float scale,
    int group_size
) {
    constexpr int NUM_WARPS = HEAD_DIM / 32;

    const int batch_idx   = blockIdx.x;
    const int q_head_idx  = blockIdx.y;
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

    float m_i = -1e9f;
    float l_i = 0.0f;
    float acc = 0.0f;

    for (int p = 0; p < num_pages; p++) {
        const int  page  = kv_indices[seq_start + p];
        const bool last  = (p == num_pages - 1);
        const int  valid = last ? last_len : PAGE_SIZE;

        // Step 1: load K page → shared memory
        {
            const __half* k_base =
                KV + page * stride_kvp + 0 * stride_kvr + kv_head_idx * stride_kvh;
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; tok++)
                sm.kv[tok * HEAD_DIM + tid] = (tok < valid)
                    ? __half2float(k_base[tok * stride_kvs + tid]) : 0.0f;
        }
        __syncthreads();

        // Step 2: QK scores via warp reduce + cross-warp reduce
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

        // Step 3: online softmax stats (registers only)
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

        // Step 4: load V page (reuse K tile shared memory)
        {
            const __half* v_base =
                KV + page * stride_kvp + 1 * stride_kvr + kv_head_idx * stride_kvh;
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; tok++)
                sm.kv[tok * HEAD_DIM + tid] = (tok < valid)
                    ? __half2float(v_base[tok * stride_kvs + tid]) : 0.0f;
        }
        __syncthreads();

        // Step 5: accumulate weighted V
        float acc_block = 0.0f;
        #pragma unroll
        for (int tok = 0; tok < PAGE_SIZE; tok++)
            acc_block += exp_weights[tok] * sm.kv[tok * HEAD_DIM + tid];

        // Step 6: merge into global online-softmax state
        const float m_new = fmaxf(m_i, m_block);
        const float alpha  = expf(m_i     - m_new);
        const float beta   = expf(m_block - m_new);
        l_i = alpha * l_i + beta * l_block;
        acc = alpha * acc + beta * acc_block;
        m_i = m_new;
    }

    O[batch_idx * stride_ob + q_head_idx * stride_oh + tid] = __float2half(acc / l_i);
}

// ── Host-side launcher ────────────────────────────────────────────────────

template<int HEAD_DIM>
static void launch_naive_hd(
    const __half* q_ptr, const __half* kv_ptr,
    const int* indptr, const int* indices, const int* last_len,
    __half* o_ptr,
    int stride_qb, int stride_qh,
    int stride_kvp, int stride_kvr, int stride_kvs, int stride_kvh,
    int stride_ob, int stride_oh,
    float scale, int group_size,
    int batch, int num_q_heads
) {
    constexpr int PAGE_SIZE = 16;
    const int smem = SmemLayout<HEAD_DIM, PAGE_SIZE>::BYTES;
    decode_attn_naive_kernel<HEAD_DIM, PAGE_SIZE>
        <<<dim3(batch, num_q_heads), HEAD_DIM, smem>>>(
            q_ptr, kv_ptr, indptr, indices, last_len, o_ptr,
            stride_qb, stride_qh,
            stride_kvp, stride_kvr, stride_kvs, stride_kvh,
            stride_ob, stride_oh,
            scale, group_size
        );
}

// ── PyTorch extension entry point ─────────────────────────────────────────

torch::Tensor decode_attention_naive_cuda(
    torch::Tensor q,
    torch::Tensor kv_data,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    torch::Tensor kv_last_page_len
) {
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(kv_data.size(2) == 16, "page_size must be 16");

    const int batch        = q.size(0);
    const int num_q_heads  = q.size(1);
    const int head_dim     = q.size(2);
    const int num_kv_heads = kv_data.size(3);
    const int group_size   = num_q_heads / num_kv_heads;
    const float scale      = 1.0f / sqrtf(static_cast<float>(head_dim));

    auto out    = torch::empty_like(q);
    auto q_ptr  = reinterpret_cast<const __half*>(q.data_ptr());
    auto kv_ptr = reinterpret_cast<const __half*>(kv_data.data_ptr());
    auto o_ptr  = reinterpret_cast<__half*>(out.data_ptr());

#define ARGS q_ptr, kv_ptr,                                  \
    kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(),   \
    kv_last_page_len.data_ptr<int>(), o_ptr,                 \
    (int)q.stride(0),       (int)q.stride(1),                \
    (int)kv_data.stride(0), (int)kv_data.stride(1),          \
    (int)kv_data.stride(2), (int)kv_data.stride(3),          \
    (int)out.stride(0),     (int)out.stride(1),              \
    scale, group_size, batch, num_q_heads

    if      (head_dim ==  64) launch_naive_hd< 64>(ARGS);
    else if (head_dim == 128) launch_naive_hd<128>(ARGS);
    else if (head_dim == 256) launch_naive_hd<256>(ARGS);
    else TORCH_CHECK(false, "head_dim must be 64, 128, or 256; got ", head_dim);
#undef ARGS

    return out;
}

// ── pybind11 module ───────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA decode attention — naive single-block kernel";
    m.def("decode_attention_naive",
          &decode_attention_naive_cuda,
          "Naive decode attention: one CUDA block per (batch, q_head)",
          py::arg("q"), py::arg("kv_data"), py::arg("kv_indptr"),
          py::arg("kv_indices"), py::arg("kv_last_page_len"));
}

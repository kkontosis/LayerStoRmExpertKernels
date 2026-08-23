// MoE token permutation and unpermutation CUDA kernels.
//
// Adapted from vLLM csrc/moe/permute_unpermute_kernels/
// moe_permute_unpermute_kernel.cu/.inl (Apache-2.0).
// Original: expandInputRowsKernel, finalizeMoeRoutingKernel,
// CubKeyValueSorter, computeExpertFirstTokenOffset.
//
// Stripped PyTorch/c10 dependencies: raw CUDA pointers + explicit stream.
// Added sentinel handling for adaptive gating (index == -1 → num_experts).

#include "smxx/permute/moe_permute.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// ── CUB radix sort helper ──────────────────────────────────────────────────

int experts_to_bits(int num_experts) {
    // Max value: num_experts (sentinel) → need log2(num_experts) + 1 bits.
    // Using 2*num_experts - 1 for safe margin (from vLLM).
    return static_cast<int>(std::log2(2.0 * num_experts + 1)) + 1;
}

size_t cub_sort_workspace_size(size_t num_pairs, int num_experts) {
    int num_bits = experts_to_bits(num_experts);
    size_t required = 0;
    int* null_int = nullptr;
    cub::DeviceRadixSort::SortPairs(nullptr, required, null_int, null_int,
                                     null_int, null_int, num_pairs, 0,
                                     num_bits);
    return std::max(required, size_t(1));
}

// ── Sentinel remap kernel ──────────────────────────────────────────────────
// Remap topk_indices == -1 (adaptive gating sentinel) to num_experts so they
// sort to the end during radix sort. Also build source row indices.

__global__ void remap_sentinels_and_build_source_rows(
    int32_t* remapped_indices,      // [num_tokens * topk] output
    int32_t* source_rows,           // [num_tokens * topk] output
    const int32_t* topk_indices,    // [num_tokens, topk] input
    int num_tokens, int topk, int num_experts) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * topk;
    if (idx >= total) return;

    int32_t expert_id = topk_indices[idx];
    // Remap sentinel (-1) to num_experts so it sorts to end
    remapped_indices[idx] = (expert_id < 0) ? num_experts : expert_id;
    source_rows[idx] = idx;  // original expanded index
}

// ── Expert offset computation ──────────────────────────────────────────────
// Binary search to find first occurrence of each expert ID in sorted array.

template <typename T>
__device__ inline int32_t find_total_less_than(const T* sorted, int32_t len,
                                                T target) {
    int32_t low = 0, high = len - 1, result = 0;
    while (low <= high) {
        int32_t mid = (low + high) / 2;
        if (sorted[mid] < target) {
            result = mid + 1;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return result;
}

__global__ void compute_expert_offsets_kernel(
    const int32_t* sorted_experts, int32_t total_items,
    int num_experts, int32_t* expert_offsets) {

    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    // We compute [0..num_experts] inclusive (num_experts+1 entries)
    if (expert > num_experts) return;

    expert_offsets[expert] = find_total_less_than(
        sorted_experts, total_items, static_cast<int32_t>(expert));
}

// ── Expand input rows (permute) ────────────────────────────────────────────
// One block per destination row. 256 threads. 128-bit vectorized copy.
// Adapted from vLLM expandInputRowsKernel.

__global__ void expand_input_rows_bf16(
    const __nv_bfloat16* __restrict__ unpermuted_input,
    __nv_bfloat16* __restrict__ permuted_output,
    const int32_t* __restrict__ sorted_rows,      // dest → source expanded row
    int32_t* __restrict__ src_to_dest_map,         // reverse: source → dest
    int32_t* __restrict__ permuted_idx,            // forward: dest → source
    int32_t cols, int32_t topk) {

    int32_t dest_row = blockIdx.x;
    int32_t source_expanded_row = sorted_rows[dest_row];

    if (threadIdx.x == 0) {
        src_to_dest_map[source_expanded_row] = dest_row;
        permuted_idx[dest_row] = source_expanded_row;
    }

    // Source token index = source_expanded_row / topk
    int32_t source_token = source_expanded_row / topk;

    // 128-bit vectorized copy: 8 BF16 elements per load
    constexpr int ELEMS_PER_VEC = 8;  // 128 bits / 16 bits
    using Vec = int4;  // 128-bit type

    int32_t num_vecs = cols / ELEMS_PER_VEC;
    auto const* src = reinterpret_cast<const Vec*>(
        unpermuted_input + static_cast<int64_t>(source_token) * cols);
    auto* dst = reinterpret_cast<Vec*>(
        permuted_output + static_cast<int64_t>(dest_row) * cols);

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        dst[i] = __ldg(&src[i]);
    }

    // Handle remainder (cols not divisible by 8)
    int remainder_start = num_vecs * ELEMS_PER_VEC;
    for (int i = remainder_start + threadIdx.x; i < cols; i += blockDim.x) {
        permuted_output[static_cast<int64_t>(dest_row) * cols + i] =
            __ldg(&unpermuted_input[static_cast<int64_t>(source_token) * cols + i]);
    }
}

__global__ void expand_input_rows_fp16(
    const __half* __restrict__ unpermuted_input,
    __half* __restrict__ permuted_output,
    const int32_t* __restrict__ sorted_rows,
    int32_t* __restrict__ src_to_dest_map,
    int32_t* __restrict__ permuted_idx,
    int32_t cols, int32_t topk) {

    int32_t dest_row = blockIdx.x;
    int32_t source_expanded_row = sorted_rows[dest_row];

    if (threadIdx.x == 0) {
        src_to_dest_map[source_expanded_row] = dest_row;
        permuted_idx[dest_row] = source_expanded_row;
    }

    int32_t source_token = source_expanded_row / topk;

    constexpr int ELEMS_PER_VEC = 8;
    using Vec = int4;

    int32_t num_vecs = cols / ELEMS_PER_VEC;
    auto const* src = reinterpret_cast<const Vec*>(
        unpermuted_input + static_cast<int64_t>(source_token) * cols);
    auto* dst = reinterpret_cast<Vec*>(
        permuted_output + static_cast<int64_t>(dest_row) * cols);

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        dst[i] = __ldg(&src[i]);
    }

    int remainder_start = num_vecs * ELEMS_PER_VEC;
    for (int i = remainder_start + threadIdx.x; i < cols; i += blockDim.x) {
        permuted_output[static_cast<int64_t>(dest_row) * cols + i] =
            __ldg(&unpermuted_input[static_cast<int64_t>(source_token) * cols + i]);
    }
}

// ── Finalize MoE routing (unpermute) ───────────────────────────────────────
// K-way weighted reduction back to original token order.
// One block per original token. 256 threads. Vectorized I/O.
// Adapted from vLLM finalizeMoeRoutingKernel.

__global__ void finalize_moe_routing_bf16(
    const __nv_bfloat16* __restrict__ expanded_permuted_rows,
    __nv_bfloat16* __restrict__ reduced_output,
    const float* __restrict__ scales,
    const int32_t* __restrict__ src_to_dest_map,
    int32_t cols, int32_t topk) {

    int32_t original_row = blockIdx.x;

    // Prefetch topk dest rows + weights into registers (avoids repeated
    // global loads in the inner loop; topk <= 8 fits in registers).
    int32_t dest_rows[8];
    float   weights_r[8];
    for (int k = 0; k < topk; k++) {
        int32_t esr = original_row * topk + k;
        dest_rows[k] = __ldg(&src_to_dest_map[esr]);
        weights_r[k] = __ldg(&scales[esr]);
    }

    // 128-bit vectorized: 8 BF16 per vector
    constexpr int ELEMS_PER_VEC = 8;
    int32_t num_vecs = cols / ELEMS_PER_VEC;

    auto* out_row = reinterpret_cast<int4*>(
        reduced_output + static_cast<int64_t>(original_row) * cols);

    for (int vec_idx = threadIdx.x; vec_idx < num_vecs; vec_idx += blockDim.x) {
        float accum[ELEMS_PER_VEC];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_VEC; e++) accum[e] = 0.0f;

        for (int k = 0; k < topk; k++) {
            auto const* src_vec = reinterpret_cast<const int4*>(
                expanded_permuted_rows + static_cast<int64_t>(dest_rows[k]) * cols);
            int4 loaded = __ldg(&src_vec[vec_idx]);

            __nv_bfloat16* elems = reinterpret_cast<__nv_bfloat16*>(&loaded);
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_VEC; e++) {
                accum[e] += weights_r[k] * __bfloat162float(elems[e]);
            }
        }

        // Convert back to BF16 and store
        int4 result;
        __nv_bfloat16* result_elems = reinterpret_cast<__nv_bfloat16*>(&result);
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_VEC; e++) {
            result_elems[e] = __float2bfloat16(accum[e]);
        }
        out_row[vec_idx] = result;
    }

    // Handle remainder
    int remainder_start = num_vecs * ELEMS_PER_VEC;
    for (int i = remainder_start + threadIdx.x; i < cols; i += blockDim.x) {
        float accum = 0.0f;
        for (int k = 0; k < topk; k++) {
            accum += weights_r[k] * __bfloat162float(
                __ldg(&expanded_permuted_rows[static_cast<int64_t>(dest_rows[k]) * cols + i]));
        }
        reduced_output[static_cast<int64_t>(original_row) * cols + i] =
            __float2bfloat16(accum);
    }
}

__global__ void finalize_moe_routing_fp16(
    const __half* __restrict__ expanded_permuted_rows,
    __half* __restrict__ reduced_output,
    const float* __restrict__ scales,
    const int32_t* __restrict__ src_to_dest_map,
    int32_t cols, int32_t topk) {

    int32_t original_row = blockIdx.x;

    // Prefetch topk dest rows + weights into registers
    int32_t dest_rows[8];
    float   weights_r[8];
    for (int k = 0; k < topk; k++) {
        int32_t esr = original_row * topk + k;
        dest_rows[k] = __ldg(&src_to_dest_map[esr]);
        weights_r[k] = __ldg(&scales[esr]);
    }

    constexpr int ELEMS_PER_VEC = 8;
    int32_t num_vecs = cols / ELEMS_PER_VEC;

    auto* out_row = reinterpret_cast<int4*>(
        reduced_output + static_cast<int64_t>(original_row) * cols);

    for (int vec_idx = threadIdx.x; vec_idx < num_vecs; vec_idx += blockDim.x) {
        float accum[ELEMS_PER_VEC];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_VEC; e++) accum[e] = 0.0f;

        for (int k = 0; k < topk; k++) {
            auto const* src_vec = reinterpret_cast<const int4*>(
                expanded_permuted_rows + static_cast<int64_t>(dest_rows[k]) * cols);
            int4 loaded = __ldg(&src_vec[vec_idx]);

            __half* elems = reinterpret_cast<__half*>(&loaded);
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_VEC; e++) {
                accum[e] += weights_r[k] * __half2float(elems[e]);
            }
        }

        int4 result;
        __half* result_elems = reinterpret_cast<__half*>(&result);
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_VEC; e++) {
            result_elems[e] = __float2half(accum[e]);
        }
        out_row[vec_idx] = result;
    }

    int remainder_start = num_vecs * ELEMS_PER_VEC;
    for (int i = remainder_start + threadIdx.x; i < cols; i += blockDim.x) {
        float accum = 0.0f;
        for (int k = 0; k < topk; k++) {
            accum += weights_r[k] * __half2float(
                __ldg(&expanded_permuted_rows[static_cast<int64_t>(dest_rows[k]) * cols + i]));
        }
        reduced_output[static_cast<int64_t>(original_row) * cols + i] =
            __float2half(accum);
    }
}

// ── Finalize MoE routing (unpermute) → FP32 PER-SLOT output ────────────────
// DET-REDUCE Phase 1b (placement-INVARIANT EP combine, canonical fixed-slot
// order). Empirically, summing each GPU's resident-subset partial in fp32 and
// adding the two fp32 partials is NOT placement-invariant: fp32 is still
// non-associative across a CHANGING partition (~1e-7 residual), which over the
// 100-token autoregressive keeper re-forks the chaotic near-tie trajectory (the
// fp32-partial A/B did NOT collapse the drift — it re-randomized it). The
// bit-placement-invariant combine instead keeps the per-token K expert
// contributions SEPARATE (one fp32 value per topk slot), so the cross-GPU
// reduce is a GATHER (each slot is contributed by exactly one GPU; non-resident
// slots write 0), and the final sum is taken in a FIXED global slot order
// 0..K-1 — independent of which GPU computed each expert. The per-slot value
// c_k = weight_k * expert_out_k is GPU-independent (the expert FFN is
// deterministic per-GPU on identical SM120 HW), so the whole combine is
// bit-identical regardless of expert placement.
//
// This kernel writes, per original token, K contiguous fp32 rows
// [token, slot, hidden]; the cross-GPU fp32 allreduce gathers slots, then
// reduce_slots_fp32_to_bf16 sums over the K slots in fixed order and rounds to
// bf16 once. Output layout: reduced_output[(token*topk + k) * cols + h].
__global__ void finalize_moe_routing_bf16_to_fp32_perslot(
    const __nv_bfloat16* __restrict__ expanded_permuted_rows,
    float* __restrict__ perslot_output,
    const float* __restrict__ scales,
    const int32_t* __restrict__ src_to_dest_map,
    int32_t cols, int32_t topk) {

    int32_t original_row = blockIdx.x;

    int32_t dest_rows[8];
    float   weights_r[8];
    for (int k = 0; k < topk; k++) {
        int32_t esr = original_row * topk + k;
        dest_rows[k] = __ldg(&src_to_dest_map[esr]);
        weights_r[k] = __ldg(&scales[esr]);
    }

    // Input is 128-bit vectorized (8 BF16 / vec); output 8 FP32 = 2x float4.
    constexpr int ELEMS_PER_VEC = 8;
    int32_t num_vecs = cols / ELEMS_PER_VEC;

    for (int k = 0; k < topk; k++) {
        auto const* src_vec = reinterpret_cast<const int4*>(
            expanded_permuted_rows + static_cast<int64_t>(dest_rows[k]) * cols);
        float* out_base = perslot_output
            + (static_cast<int64_t>(original_row) * topk + k) * cols;
        const float w = weights_r[k];

        for (int vec_idx = threadIdx.x; vec_idx < num_vecs; vec_idx += blockDim.x) {
            int4 loaded = __ldg(&src_vec[vec_idx]);
            __nv_bfloat16* elems = reinterpret_cast<__nv_bfloat16*>(&loaded);
            float4 r0{w * __bfloat162float(elems[0]), w * __bfloat162float(elems[1]),
                      w * __bfloat162float(elems[2]), w * __bfloat162float(elems[3])};
            float4 r1{w * __bfloat162float(elems[4]), w * __bfloat162float(elems[5]),
                      w * __bfloat162float(elems[6]), w * __bfloat162float(elems[7])};
            auto* out_vec = reinterpret_cast<float4*>(
                out_base + static_cast<int64_t>(vec_idx) * ELEMS_PER_VEC);
            out_vec[0] = r0;
            out_vec[1] = r1;
        }
        int remainder_start = num_vecs * ELEMS_PER_VEC;
        for (int i = remainder_start + threadIdx.x; i < cols; i += blockDim.x) {
            out_base[i] = w * __bfloat162float(
                __ldg(&expanded_permuted_rows[static_cast<int64_t>(dest_rows[k]) * cols + i]));
        }
    }
}

// DET-REDUCE Phase 1b BF16-PAYLOAD variant of finalize_moe_routing_bf16_to_fp32_perslot.
// Identical structure and placement-invariance contract, but the per-slot
// contribution c_k = weight_k * expert_out_k is rounded to BF16 once and stored
// as BF16 (half the gather bytes vs fp32). Matches the vLLM/llama.cpp convention
// (fp32 math, 16-bit payload). Placement-invariance is preserved because it comes
// from the fixed SLOT ORDER, not the payload dtype: c_k is GPU-independent (the
// expert FFN is deterministic per-GPU on identical SM120 HW, the fp32 product is
// rounded to bf16 deterministically regardless of which GPU computes it), the
// cross-GPU gather is order-preserving (each slot filled by exactly one GPU; the
// bf16 SUM-allreduce is exact since 0+x=x), and the final reduce sums the K slots
// in fixed order 0..K-1 in fp32. Output layout: perslot_output[(token*topk+k)*cols+h] (bf16).
__global__ void finalize_moe_routing_bf16_to_bf16_perslot(
    const __nv_bfloat16* __restrict__ expanded_permuted_rows,
    __nv_bfloat16* __restrict__ perslot_output,
    const float* __restrict__ scales,
    const int32_t* __restrict__ src_to_dest_map,
    int32_t cols, int32_t topk) {

    int32_t original_row = blockIdx.x;

    int32_t dest_rows[8];
    float   weights_r[8];
    for (int k = 0; k < topk; k++) {
        int32_t esr = original_row * topk + k;
        dest_rows[k] = __ldg(&src_to_dest_map[esr]);
        weights_r[k] = __ldg(&scales[esr]);
    }

    // Input and output are both 128-bit vectorized (8 BF16 / int4).
    constexpr int ELEMS_PER_VEC = 8;
    int32_t num_vecs = cols / ELEMS_PER_VEC;

    for (int k = 0; k < topk; k++) {
        auto const* src_vec = reinterpret_cast<const int4*>(
            expanded_permuted_rows + static_cast<int64_t>(dest_rows[k]) * cols);
        __nv_bfloat16* out_base = perslot_output
            + (static_cast<int64_t>(original_row) * topk + k) * cols;
        auto* out_vec = reinterpret_cast<int4*>(out_base);
        const float w = weights_r[k];

        for (int vec_idx = threadIdx.x; vec_idx < num_vecs; vec_idx += blockDim.x) {
            int4 loaded = __ldg(&src_vec[vec_idx]);
            __nv_bfloat16* elems = reinterpret_cast<__nv_bfloat16*>(&loaded);
            int4 stored;
            __nv_bfloat16* se = reinterpret_cast<__nv_bfloat16*>(&stored);
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_VEC; e++)
                se[e] = __float2bfloat16(w * __bfloat162float(elems[e]));
            out_vec[vec_idx] = stored;
        }
        int remainder_start = num_vecs * ELEMS_PER_VEC;
        for (int i = remainder_start + threadIdx.x; i < cols; i += blockDim.x) {
            out_base[i] = __float2bfloat16(w * __bfloat162float(
                __ldg(&expanded_permuted_rows[static_cast<int64_t>(dest_rows[k]) * cols + i])));
        }
    }
}

// Fixed-order K-slot reduce → BF16. Sums the topk per-slot fp32 contributions in
// canonical slot order 0..topk-1 (placement-invariant) and rounds to bf16 once
// (round-to-nearest, matching __float2bfloat16 in the bf16 unpermute path).
// in_perslot layout: [num_tokens, topk, cols] fp32; out: [num_tokens, cols] bf16.
__global__ void reduce_slots_fp32_to_bf16_kernel(
    const float* __restrict__ in_perslot, __nv_bfloat16* __restrict__ out,
    int32_t num_tokens, int32_t topk, int32_t cols) {
    int64_t total = static_cast<int64_t>(num_tokens) * cols;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t t = idx / cols;
        int64_t h = idx - t * cols;
        const float* base = in_perslot + t * topk * cols + h;
        float acc = 0.0f;
        for (int k = 0; k < topk; k++) acc += base[static_cast<int64_t>(k) * cols];
        out[idx] = __float2bfloat16(acc);
    }
}

// DET-REDUCE Phase 1b BF16-PAYLOAD reduce: same fixed-order K-slot reduce as
// reduce_slots_fp32_to_bf16_kernel, but the gathered per-slot input is BF16. Each
// slot is widened to fp32 and accumulated in fixed slot order 0..topk-1 (fp32
// math), then rounded to bf16 ONCE — matching the vLLM/llama.cpp convention.
// in_perslot layout: [num_tokens, topk, cols] bf16; out: [num_tokens, cols] bf16.
__global__ void reduce_slots_bf16_to_bf16_kernel(
    const __nv_bfloat16* __restrict__ in_perslot, __nv_bfloat16* __restrict__ out,
    int32_t num_tokens, int32_t topk, int32_t cols) {
    int64_t total = static_cast<int64_t>(num_tokens) * cols;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t t = idx / cols;
        int64_t h = idx - t * cols;
        const __nv_bfloat16* base = in_perslot + t * topk * cols + h;
        float acc = 0.0f;
        for (int k = 0; k < topk; k++)
            acc += __bfloat162float(base[static_cast<int64_t>(k) * cols]);
        out[idx] = __float2bfloat16(acc);
    }
}

// ── Workspace layout ──────────────────────────────────────────────────────
// All sub-allocations are contiguous in a single device buffer.

struct PermuteWorkspaceLayout {
    size_t remapped_indices_offset;   // int32[expanded]
    size_t source_rows_offset;        // int32[expanded]
    size_t sorted_experts_offset;     // int32[expanded] (CUB output keys)
    size_t sorted_rows_offset;        // int32[expanded] (CUB output values)
    size_t cub_workspace_offset;      // CUB temp storage
    size_t total_bytes;
};

PermuteWorkspaceLayout compute_layout(int max_tokens, int max_topk,
                                       int num_experts) {
    int64_t expanded = static_cast<int64_t>(max_tokens) * max_topk;
    size_t int_array_bytes = static_cast<size_t>(expanded) * sizeof(int32_t);
    size_t cub_bytes = cub_sort_workspace_size(expanded, num_experts + 1);
    // Pad each sub-allocation to 256 bytes for alignment
    auto align256 = [](size_t x) { return (x + 255) & ~size_t(255); };

    PermuteWorkspaceLayout layout;
    size_t offset = 0;
    layout.remapped_indices_offset = offset;  offset += align256(int_array_bytes);
    layout.source_rows_offset      = offset;  offset += align256(int_array_bytes);
    layout.sorted_experts_offset   = offset;  offset += align256(int_array_bytes);
    layout.sorted_rows_offset      = offset;  offset += align256(int_array_bytes);
    layout.cub_workspace_offset    = offset;  offset += align256(cub_bytes);
    layout.total_bytes = offset;
    return layout;
}

}  // anonymous namespace

// ── Public API ─────────────────────────────────────────────────────────────

namespace layerstorm::compute {

size_t query_moe_permute_workspace_size(int max_tokens, int max_topk,
                                         int num_experts) {
    return compute_layout(max_tokens, max_topk, num_experts).total_bytes;
}

void launch_moe_permute(
    void* permuted_input,
    int32_t* expert_offsets,
    int32_t* src_to_dest_map,
    int32_t* permuted_idx,
    const void* hidden_states,
    const int32_t* topk_indices,
    int num_tokens,
    int topk,
    int hidden_dim,
    int num_experts,
    int elem_size_bytes,
    void* workspace,
    void* stream) {

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    int64_t expanded = static_cast<int64_t>(num_tokens) * topk;
    auto layout = compute_layout(num_tokens, topk, num_experts);

    auto* ws = static_cast<char*>(workspace);
    auto* remapped = reinterpret_cast<int32_t*>(ws + layout.remapped_indices_offset);
    auto* source_rows = reinterpret_cast<int32_t*>(ws + layout.source_rows_offset);
    auto* sorted_experts = reinterpret_cast<int32_t*>(ws + layout.sorted_experts_offset);
    auto* sorted_rows = reinterpret_cast<int32_t*>(ws + layout.sorted_rows_offset);
    void* cub_ws = ws + layout.cub_workspace_offset;

    // Step 1: Remap sentinels (-1 → num_experts) and build source row indices
    {
        int threads = 256;
        int blocks = (static_cast<int>(expanded) + threads - 1) / threads;
        remap_sentinels_and_build_source_rows<<<blocks, threads, 0, cuda_stream>>>(
            remapped, source_rows, topk_indices,
            num_tokens, topk, num_experts);
    }

    // Step 2: CUB radix sort — group tokens by expert ID
    {
        // Sort keys=remapped expert IDs, values=source row indices
        int num_bits = experts_to_bits(num_experts + 1);
        size_t cub_bytes = cub_sort_workspace_size(expanded, num_experts + 1);
        cub::DeviceRadixSort::SortPairs(
            cub_ws, cub_bytes,
            remapped, sorted_experts,
            source_rows, sorted_rows,
            expanded, 0, num_bits, cuda_stream);
    }

    // Step 3: Compute expert offsets via binary search
    {
        int num_entries = num_experts + 1;
        int threads = std::min(1024, num_entries);
        int blocks = (num_entries + threads - 1) / threads;
        compute_expert_offsets_kernel<<<blocks, threads, 0, cuda_stream>>>(
            sorted_experts, static_cast<int32_t>(expanded),
            num_experts, expert_offsets);
    }

    // Step 4: Expand input rows — permute tokens into expert-grouped layout
    {
        int blocks = static_cast<int>(expanded);
        int threads = 256;
        if (elem_size_bytes == 2) {
            // BF16 or FP16: detect based on elem_size_bytes. We use BF16 as
            // the default path since both are 2 bytes and the copy is bitwise.
            expand_input_rows_bf16<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __nv_bfloat16*>(hidden_states),
                static_cast<__nv_bfloat16*>(permuted_input),
                sorted_rows, src_to_dest_map, permuted_idx,
                hidden_dim, topk);
        } else if (elem_size_bytes == 4) {
            // FP32 fallback: treat as 2× BF16 (bitwise copy is fine)
            expand_input_rows_bf16<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __nv_bfloat16*>(hidden_states),
                static_cast<__nv_bfloat16*>(permuted_input),
                sorted_rows, src_to_dest_map, permuted_idx,
                hidden_dim * 2, topk);
        } else {
            throw std::runtime_error(
                "moe_permute: unsupported elem_size_bytes=" +
                std::to_string(elem_size_bytes));
        }
    }
}

void launch_moe_unpermute(
    void* output,
    const void* permuted_output,
    const float* topk_weights,
    const int32_t* src_to_dest_map,
    int num_tokens,
    int topk,
    int hidden_dim,
    int elem_size_bytes,
    void* stream,
    bool fp32_output) {

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    int blocks = num_tokens;
    int threads = 256;

    if (elem_size_bytes == 2) {
        if (fp32_output) {
            // DET-REDUCE Phase 1b (canonical): bf16 expert outputs → fp32
            // PER-SLOT (no cross-slot sum). `output` must be an FP32 buffer
            // [num_tokens, topk, hidden_dim]. Combined later by a fixed-order
            // K-slot reduce (launch_moe_combine_reduce_slots_fp32_to_bf16).
            finalize_moe_routing_bf16_to_fp32_perslot<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __nv_bfloat16*>(permuted_output),
                static_cast<float*>(output),
                topk_weights, src_to_dest_map,
                hidden_dim, topk);
        } else {
            // Default: BF16 in, BF16 out (same kernel works for FP16, bitwise).
            finalize_moe_routing_bf16<<<blocks, threads, 0, cuda_stream>>>(
                static_cast<const __nv_bfloat16*>(permuted_output),
                static_cast<__nv_bfloat16*>(output),
                topk_weights, src_to_dest_map,
                hidden_dim, topk);
        }
    } else {
        throw std::runtime_error(
            "moe_unpermute: unsupported elem_size_bytes=" +
            std::to_string(elem_size_bytes));
    }
}

void launch_moe_unpermute_bf16_perslot(
    void* output,
    const void* permuted_output,
    const float* topk_weights,
    const int32_t* src_to_dest_map,
    int num_tokens,
    int topk,
    int hidden_dim,
    int elem_size_bytes,
    void* stream) {

    if (elem_size_bytes != 2)
        throw std::runtime_error(
            "moe_unpermute_bf16_perslot: unsupported elem_size_bytes=" +
            std::to_string(elem_size_bytes));

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    int blocks = num_tokens;
    int threads = 256;
    // DET-REDUCE Phase 1b (canonical, BF16 payload): bf16 expert outputs →
    // bf16 PER-SLOT (no cross-slot sum). `output` must be a BF16 buffer
    // [num_tokens, topk, hidden_dim]. Combined later by a fixed-order K-slot
    // reduce (launch_moe_combine_reduce_slots_bf16_to_bf16).
    finalize_moe_routing_bf16_to_bf16_perslot<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const __nv_bfloat16*>(permuted_output),
        static_cast<__nv_bfloat16*>(output),
        topk_weights, src_to_dest_map,
        hidden_dim, topk);
}

void launch_moe_combine_reduce_slots_fp32_to_bf16(
    void* output_bf16,
    const void* input_perslot_fp32,
    int num_tokens,
    int topk,
    int hidden_dim,
    void* stream) {

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    int64_t n = static_cast<int64_t>(num_tokens) * hidden_dim;
    if (n <= 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;  // grid-stride loop covers the rest
    reduce_slots_fp32_to_bf16_kernel<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const float*>(input_perslot_fp32),
        static_cast<__nv_bfloat16*>(output_bf16), num_tokens, topk, hidden_dim);
}

void launch_moe_combine_reduce_slots_bf16_to_bf16(
    void* output_bf16,
    const void* input_perslot_bf16,
    int num_tokens,
    int topk,
    int hidden_dim,
    void* stream) {

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    int64_t n = static_cast<int64_t>(num_tokens) * hidden_dim;
    if (n <= 0) return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;  // grid-stride loop covers the rest
    reduce_slots_bf16_to_bf16_kernel<<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const __nv_bfloat16*>(input_perslot_bf16),
        static_cast<__nv_bfloat16*>(output_bf16), num_tokens, topk, hidden_dim);
}

}  // namespace layerstorm::compute

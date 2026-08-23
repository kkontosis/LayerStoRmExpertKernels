// Top-K gating CUDA kernels for LayerStoRm MoE routing.
// Adapted from TRT-LLM RoutingKernelTopK.cuh and vLLM grouped_topk_kernels.cu
// (both Apache-2.0).
//
// Two kernel variants:
//   simple_topk_kernel  — flat top-K for n_group=1 (GLM-5, Kimi K2.5, V4)
//   grouped_topk_kernel — two-level grouped routing for n_group>1 (DeepSeek V3.2)
//
// Scoring functions: sigmoid (tanh identity), sqrtsoftplus (V4).
// Bias applied for selection only; weights use unbiased scores.
//
// SM120 improvements over previous SM100-era implementation:
//   · Warp-level packed (value, idx) top-K selection (TRT-LLM RoutingKernelTopK.cuh)
//     replaces iterative shared-memory block argmax.  For topk=8 this removes
//     8 block-level barriers from the critical path (from 12 to 4 total syncs).
//   · Selection done entirely in warp registers via cooperative_groups::reduce
//     — no shared memory traffic in the final top-K step.

#include "sm120/gating/topk_gating.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>

namespace cg = cooperative_groups;

namespace layerstorm::compute {

// ── Device helpers ──────────────────────────────────────────────────────────

/// Numerically stable sigmoid via tanh identity.
__device__ __forceinline__ float stable_sigmoid(float x) {
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

/// sqrt(softplus(x)) = sqrt(log(1 + exp(x))).
/// For x > 20: softplus(x) = x + log(1 + exp(-x)) to avoid exp(x) overflow.
__device__ __forceinline__ float stable_sqrtsoftplus(float x) {
    float sp = (x > 20.0f) ? x + logf(1.0f + expf(-x))
                            : logf(1.0f + expf(x));
    return sqrtf(sp);
}

/// Dispatch scoring function by enum value (passed as int to kernels).
__device__ __forceinline__ float apply_scoring(float logit, int scoring_func) {
    if (scoring_func == 0)
        return stable_sigmoid(logit);
    return stable_sqrtsoftplus(logit);
}

// ── Packed (value, idx) comparison (TRT-LLM RoutingKernelTopK.cuh) ──────────
//
// Pack float score + int32 expert index into a uint64_t so that:
//   - uint64 ordering matches float ordering (using CUB/TRT-LLM twiddle)
//   - ties broken by lower expert index
//
// This allows warp-level max reduction on uint64_t via cg::reduce().

__device__ __forceinline__ uint64_t pack_score(float val, int32_t idx) {
    uint32_t bits;
    __builtin_memcpy(&bits, &val, sizeof(float));
    // Twiddle: flip sign bit for positives, flip all bits for negatives.
    // Result: uint32_t ordering matches float ordering.
    const uint32_t sign = bits >> 31;
    bits ^= 0x80000000u | (-sign);  // -sign is 0 or 0xFFFFFFFF
    // Lower expert index → higher lower 32 bits (tie-breaking).
    return (static_cast<uint64_t>(bits) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(0x7fffffffu) -
                                 static_cast<uint32_t>(idx));
}

__device__ __forceinline__ void unpack_score(uint64_t packed, float& val,
                                             int32_t& idx) {
    uint32_t bits = static_cast<uint32_t>(packed >> 32);
    // Undo twiddle.
    const uint32_t sign = bits >> 31;
    bits ^= 0x80000000u | (sign - 1u);  // sign-1 is 0 or 0xFFFFFFFF
    __builtin_memcpy(&val, &bits, sizeof(float));
    idx = static_cast<int32_t>(0x7fffffffu -
                                static_cast<uint32_t>(packed & 0xffffffffu));
}

/// Sentinel: packed representation of (-FLT_MAX, very large idx).
/// Any valid candidate will beat this in a warp max-reduce.
__device__ __forceinline__ constexpr uint64_t packed_sentinel() {
    // bits for -FLT_MAX = 0xFF7FFFFF; twiddle(sign=1) → ~bits = 0x00800000.
    // idx_enc = 0x7FFFFFFF - 0x3FFFFFFF = 0x40000000.
    return (static_cast<uint64_t>(0x00800000u) << 32) |
           static_cast<uint64_t>(0x40000000u);
}

// ── Warp-level top-K via packed reduction ───────────────────────────────────
// After this returns: out_w[0..topk-1] and out_i[0..topk-1] are valid
// in lane 0 of the calling warp only (other lanes have stale data).
//
// packed[]:  per-lane candidate array, sorted descending on entry (best first).
//            Array is consumed in-place; callers may not reuse it.
// n_cands:   number of valid entries in packed[] (padded slots hold sentinels).
// s_scores[]: shared memory array of unbiased sigmoid scores indexed by expert.

__device__ void warp_topk_select(
    cg::thread_block_tile<32> warp, uint64_t* packed, int n_cands,
    const float* __restrict__ s_scores, float* __restrict__ out_w,
    int32_t* __restrict__ out_i, int topk, float routed_scaling_factor,
    bool renormalize) {

    float sum_renorm = 0.f;

    for (int k = 0; k < topk; ++k) {
        uint64_t warp_max = cg::reduce(warp, packed[0], cg::greater<uint64_t>{});

        if (warp.thread_rank() == 0) {
            float dummy_val;
            int32_t expert_idx;
            unpack_score(warp_max, dummy_val, expert_idx);
            const float unbiased_w = s_scores[expert_idx];
            out_i[k] = expert_idx;
            out_w[k] = unbiased_w;
            sum_renorm += unbiased_w;
        }

        // The lane that held warp_max advances its local candidate queue.
        if (packed[0] == warp_max) {
            for (int j = 0; j < n_cands - 1; ++j)
                packed[j] = packed[j + 1];
            packed[n_cands - 1] = packed_sentinel();
        }
    }

    if (renormalize && warp.thread_rank() == 0 && sum_renorm > 0.f) {
        const float scale = routed_scaling_factor / sum_renorm;
        for (int k = 0; k < topk; ++k)
            out_w[k] *= scale;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Simple Top-K Kernel (n_group = 1): GLM-5, Kimi K2.5
// ═════════════════════════════════════════════════════════════════════════════
//
// One block per token, 256 threads. All experts are candidates.
//
// Phases:
//   1. All threads compute sigmoid scores → shared memory.
//   2. Warp 0 loads candidates into registers (packed uint64_t), sorts locally,
//      then runs warp_topk_select — zero shared-memory traffic for the top-K
//      step itself.
//
// Shared memory layout:
//   s_scores[E]  — unbiased sigmoid scores (for output weights)
//   s_sel[E]     — selection scores (with bias, for top-K)

__global__ void simple_topk_kernel(
    float* __restrict__ topk_weights, int32_t* __restrict__ topk_indices,
    const float* __restrict__ logits, const float* __restrict__ bias,
    int num_tokens, int num_experts, int topk, float routed_scaling_factor,
    bool renormalize, int scoring_func) {

    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    extern __shared__ char smem[];
    float* s_scores = reinterpret_cast<float*>(smem);
    float* s_sel    = s_scores + num_experts;

    const float* token_logits =
        logits + static_cast<int64_t>(token_idx) * num_experts;

    // Phase 1: all threads compute scores (unbiased + biased).
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const float s = apply_scoring(token_logits[e], scoring_func);
        s_scores[e] = s;
        s_sel[e] = bias ? s + bias[e] : s;
    }
    __syncthreads();  // sync 1: scores visible to all

    // Phase 2: warp 0 selects top-K via packed warp-register reduction.
    // Each of the 32 lanes holds ceil(num_experts/32) candidates.
    // Maximum supported: 16 candidates/lane → up to 512 experts.
    const int warpIdx = threadIdx.x / 32;
    const int laneIdx = threadIdx.x & 31;

    if (warpIdx == 0) {
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());

        constexpr int kMaxCands = 16;  // handles up to 512 experts
        uint64_t packed[kMaxCands];
        const int n_per_lane = (num_experts + 31) / 32;

        // Load candidates.
        for (int i = 0; i < n_per_lane; ++i) {
            const int e = laneIdx + i * 32;
            const float score = (e < num_experts) ? s_sel[e] : -FLT_MAX;
            const int safe_e  = (e < num_experts) ? e : num_experts;
            packed[i] = pack_score(score, safe_e);
        }
        // Pad unused slots with sentinels.
        for (int i = n_per_lane; i < kMaxCands; ++i)
            packed[i] = packed_sentinel();

        // Sort local candidates descending (insertion sort, small n_per_lane).
        for (int i = 1; i < n_per_lane; ++i) {
            const uint64_t key = packed[i];
            int j = i - 1;
            while (j >= 0 && packed[j] < key) {
                packed[j + 1] = packed[j];
                --j;
            }
            packed[j + 1] = key;
        }

        float* out_w = topk_weights + static_cast<int64_t>(token_idx) * topk;
        int32_t* out_i =
            topk_indices + static_cast<int64_t>(token_idx) * topk;

        warp_topk_select(warp, packed, n_per_lane, s_scores, out_w, out_i,
                         topk, routed_scaling_factor, renormalize);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Grouped Top-K Kernel (n_group > 1): DeepSeek V3.2
// ═════════════════════════════════════════════════════════════════════════════
//
// Two-level routing. One block per token, one warp per group.
//
// Phase 1: Each warp finds sum-of-top-2 selection scores in its group.
// Phase 2: Thread 0 picks top-topk_group groups.
// Phase 3: Mask non-candidate experts from unselected groups.
// Phase 4: Warp 0 selects top-topk from candidates via packed warp-register
//          reduction (no shared-memory traffic, no barriers in this phase).
//
// Shared memory layout:
//   s_scores[E]           — unbiased sigmoid scores
//   s_sel[E]              — selection scores (masked after group filtering)
//   s_group_scores[G]     — per-group top-2 sum
//   s_selected_groups[G]  — indices of the topk_group selected groups

__global__ void grouped_topk_kernel(
    float* __restrict__ topk_weights, int32_t* __restrict__ topk_indices,
    const float* __restrict__ logits, const float* __restrict__ bias,
    int num_tokens, int num_experts, int n_group, int topk_group, int topk,
    float routed_scaling_factor, bool renormalize, int scoring_func) {

    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    const int experts_per_group = num_experts / n_group;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;

    extern __shared__ char smem[];
    float* s_scores        = reinterpret_cast<float*>(smem);
    float* s_sel           = s_scores + num_experts;
    float* s_group_scores  = s_sel + num_experts;
    int*   s_selected_groups =
        reinterpret_cast<int*>(s_group_scores + n_group);

    const float* token_logits =
        logits + static_cast<int64_t>(token_idx) * num_experts;

    // ── Phase 1: compute all scores ────────────────────────────────────────
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const float s = apply_scoring(token_logits[e], scoring_func);
        s_scores[e] = s;
        s_sel[e]    = bias ? s + bias[e] : s;
    }
    __syncthreads();  // sync 1

    // ── Phase 2: per-group top-2 scoring (one warp per group) ───────────────
    if (warp_id < n_group) {
        const int group_start = warp_id * experts_per_group;
        float top1 = -FLT_MAX, top2 = -FLT_MAX;
        for (int i = lane_id; i < experts_per_group; i += 32) {
            const float v = s_sel[group_start + i];
            if (v > top1) {
                top2 = top1;
                top1 = v;
            } else if (v > top2) {
                top2 = v;
            }
        }
        // Warp reduction: merge top-2 pairs across lanes.
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float o1 = __shfl_xor_sync(0xffffffff, top1, offset);
            const float o2 = __shfl_xor_sync(0xffffffff, top2, offset);
            if (top1 >= o1) {
                top2 = (top2 >= o1) ? top2 : o1;
            } else {
                top2 = (o2 >= top1) ? o2 : top1;
                top1 = o1;
            }
        }
        if (lane_id == 0)
            s_group_scores[warp_id] = top1 + top2;
    }
    __syncthreads();  // sync 2

    // ── Phase 3: thread 0 picks top-topk_group groups ───────────────────────
    if (threadIdx.x == 0) {
        for (int g = 0; g < topk_group; ++g) {
            float best   = -FLT_MAX;
            int   best_g = 0;
            for (int i = 0; i < n_group; ++i) {
                if (s_group_scores[i] > best) {
                    best   = s_group_scores[i];
                    best_g = i;
                }
            }
            s_selected_groups[g] = best_g;
            s_group_scores[best_g] = -FLT_MAX;
        }
    }
    __syncthreads();  // sync 3

    // ── Phase 4: mask non-candidate experts ─────────────────────────────────
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const int group = e / experts_per_group;
        bool selected = false;
        for (int g = 0; g < topk_group; ++g) {
            if (s_selected_groups[g] == group) {
                selected = true;
                break;
            }
        }
        if (!selected)
            s_sel[e] = -FLT_MAX;
    }
    __syncthreads();  // sync 4

    // ── Phase 5: warp 0 selects top-topk from candidates ───────────────────
    // Each lane covers ceil(experts_per_group/WarpSize) experts per selected
    // group. For V3.2 (32 experts/group, 32 lanes): exactly 1 per group chunk,
    // so topk_group=4 candidates per lane. Generalized to any config.
    if (warp_id == 0) {
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());

        // kMaxCands: topk_group groups × max experts_per_lane chunks.
        // With topk_group ≤ 4 and ceil(experts_per_group/32) ≤ 4 → max 16.
        constexpr int kMaxCands = 16;
        uint64_t packed[kMaxCands];

        const int chunks_per_group = (experts_per_group + 31) / 32;
        const int n_per_lane = topk_group * chunks_per_group;

        int slot = 0;
        for (int g = 0; g < topk_group; ++g) {
            const int grp = s_selected_groups[g];
            for (int c = 0; c < chunks_per_group; ++c) {
                const int e = grp * experts_per_group + c * 32 + lane_id;
                const bool valid =
                    (lane_id < experts_per_group - c * 32) &&
                    (e < num_experts);
                const float score = valid ? s_sel[e] : -FLT_MAX;
                const int   safe_e = valid ? e : num_experts;
                packed[slot++] = pack_score(score, safe_e);
            }
        }
        // Pad.
        for (int i = slot; i < kMaxCands; ++i)
            packed[i] = packed_sentinel();

        // Sort local candidates descending.
        for (int i = 1; i < n_per_lane; ++i) {
            const uint64_t key = packed[i];
            int j = i - 1;
            while (j >= 0 && packed[j] < key) {
                packed[j + 1] = packed[j];
                --j;
            }
            packed[j + 1] = key;
        }

        float*   out_w = topk_weights + static_cast<int64_t>(token_idx) * topk;
        int32_t* out_i = topk_indices + static_cast<int64_t>(token_idx) * topk;

        warp_topk_select(warp, packed, n_per_lane, s_scores, out_w, out_i,
                         topk, routed_scaling_factor, renormalize);
    }
}

// ── Launch dispatcher ───────────────────────────────────────────────────────

void launch_topk_gating(float* topk_weights, int32_t* topk_indices,
                        const float* logits, const float* bias,
                        const TopkGatingParams& params, void* stream) {
    if (params.num_tokens <= 0)
        return;

    if (params.num_experts <= 0)
        throw std::invalid_argument(
            "launch_topk_gating: num_experts must be > 0");
    if (params.topk <= 0 || params.topk > params.num_experts)
        throw std::invalid_argument(
            "launch_topk_gating: topk must be in [1, num_experts]");
    if (params.n_group < 1 || params.num_experts % params.n_group != 0)
        throw std::invalid_argument(
            "launch_topk_gating: num_experts must be divisible by n_group");
    if (params.n_group > 1 &&
        (params.topk_group < 1 || params.topk_group > params.n_group))
        throw std::invalid_argument(
            "launch_topk_gating: topk_group must be in [1, n_group]");

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    const int sf = static_cast<int>(params.scoring_func);

    if (params.n_group <= 1) {
        // ── Simple top-K (GLM-5, K2.5, V4) ─────────────────────────────────
        constexpr int kBlockSize = 256;

        // smem: s_scores[E] + s_sel[E]
        const size_t smem_bytes =
            static_cast<size_t>(params.num_experts) * 2 * sizeof(float);

        simple_topk_kernel<<<params.num_tokens, kBlockSize, smem_bytes,
                             cuda_stream>>>(
            topk_weights, topk_indices, logits, bias, params.num_tokens,
            params.num_experts, params.topk, params.routed_scaling_factor,
            params.renormalize, sf);
    } else {
        // ── Grouped top-K (V3.2) ─────────────────────────────────────────
        const int block_size = params.n_group * 32;  // one warp per group

        // smem: s_scores[E] + s_sel[E] + s_group_scores[G] + s_selected_groups[G]
        const size_t smem_bytes =
            static_cast<size_t>(params.num_experts) * 2 * sizeof(float) +
            static_cast<size_t>(params.n_group) * sizeof(float) +
            static_cast<size_t>(params.n_group) * sizeof(int);

        grouped_topk_kernel<<<params.num_tokens, block_size, smem_bytes,
                              cuda_stream>>>(
            topk_weights, topk_indices, logits, bias, params.num_tokens,
            params.num_experts, params.n_group, params.topk_group, params.topk,
            params.routed_scaling_factor, params.renormalize, sf);
    }
}

}  // namespace layerstorm::compute

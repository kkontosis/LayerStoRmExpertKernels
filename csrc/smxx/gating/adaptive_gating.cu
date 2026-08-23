// Adaptive gating CUDA kernel for LayerStoRm MoE routing.
// Post-processes top-K gating output to produce variable-length expert lists
// per token via cumulative routing weight threshold.
//
// Single kernel, one thread per token. K ≤ 8 fits entirely in registers.
// Insertion sort (28 comparisons worst case for K=8), then cumulative scan.
//
// Reviewed against TRT-LLM and vLLM references for SM120 (GeForce Blackwell).
// Algorithm is register-bound with no shared memory; the pure-register design
// is already optimal for SM120 — no architecture-specific changes required.

#include "smxx/gating/adaptive_gating.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace layerstorm::compute {

static constexpr int kMaxTopk = 8;

__global__ void adaptive_gating_kernel(
    float* __restrict__ out_weights, int32_t* __restrict__ out_indices,
    int32_t* __restrict__ expert_counts,
    const float* __restrict__ in_weights,
    const int32_t* __restrict__ in_indices,
    int num_tokens, int topk, float threshold) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tokens) return;

    const int offset = tid * topk;

    // Load into registers.
    float weights[kMaxTopk];
    int32_t indices[kMaxTopk];
    for (int k = 0; k < topk; ++k) {
        weights[k] = in_weights[offset + k];
        indices[k] = in_indices[offset + k];
    }

    // Insertion sort by weight descending (needed because top-K output is
    // sorted by biased selection score, not unbiased weight).
    for (int i = 1; i < topk; ++i) {
        float w = weights[i];
        int32_t idx = indices[i];
        int j = i - 1;
        while (j >= 0 && weights[j] < w) {
            weights[j + 1] = weights[j];
            indices[j + 1] = indices[j];
            --j;
        }
        weights[j + 1] = w;
        indices[j + 1] = idx;
    }

    // Compute total weight.
    float total_weight = 0.0f;
    for (int k = 0; k < topk; ++k) {
        total_weight += weights[k];
    }

    // Cumulative scan: keep experts until cumsum >= threshold * total_weight.
    // Always keep at least 1 expert.
    float target = threshold * total_weight;
    float cumsum = 0.0f;
    int count = topk;  // default: keep all
    for (int k = 0; k < topk; ++k) {
        cumsum += weights[k];
        if (cumsum >= target) {
            count = k + 1;
            break;
        }
    }
    if (count < 1) count = 1;  // safety: always keep at least 1

    // Write kept experts.
    for (int k = 0; k < count; ++k) {
        out_weights[offset + k] = weights[k];
        out_indices[offset + k] = indices[k];
    }

    // Pad remaining slots with sentinel.
    for (int k = count; k < topk; ++k) {
        out_weights[offset + k] = 0.0f;
        out_indices[offset + k] = -1;
    }

    expert_counts[tid] = count;
}

// ── Launch dispatcher ───────────────────────────────────────────────────────

void launch_adaptive_gating(float* out_weights, int32_t* out_indices,
                             int32_t* expert_counts,
                             const float* in_weights,
                             const int32_t* in_indices,
                             const AdaptiveGatingParams& params,
                             void* stream) {
    if (params.num_tokens <= 0) return;

    if (params.topk < 1 || params.topk > kMaxTopk)
        throw std::invalid_argument(
            "launch_adaptive_gating: topk must be in [1, 8]");
    if (params.threshold < 0.0f || params.threshold > 1.0f)
        throw std::invalid_argument(
            "launch_adaptive_gating: threshold must be in [0.0, 1.0]");

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    constexpr int kBlockSize = 256;
    const int grid = (params.num_tokens + kBlockSize - 1) / kBlockSize;

    adaptive_gating_kernel<<<grid, kBlockSize, 0, cuda_stream>>>(
        out_weights, out_indices, expert_counts, in_weights, in_indices,
        params.num_tokens, params.topk, params.threshold);
}

}  // namespace layerstorm::compute

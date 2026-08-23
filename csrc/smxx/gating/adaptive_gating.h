// Adaptive gating CUDA kernel for LayerStoRm MoE routing.
// Post-processes top-K gating output (#29) to produce variable-length expert
// lists per token using a cumulative routing weight threshold.
//
// Model-agnostic: operates on post-gating output regardless of how top-K was
// computed (grouped or simple). Supports INV-0.2 (no fixed expert count).

#pragma once

#include <cstdint>

namespace layerstorm::compute {

/// Parameters for adaptive gating kernel launch.
struct AdaptiveGatingParams {
    int num_tokens;
    int topk;        // max experts per token from top-K stage (≤ 8)
    float threshold;  // cumulative weight threshold [0.0, 1.0]
};

/// Adaptive gating: threshold-based expert pruning on top-K output.
///
/// Re-sorts experts by weight descending (top-K output is sorted by biased
/// selection score, not unbiased weight), then keeps the minimum set whose
/// cumulative weight >= threshold * total_weight. Always keeps at least 1.
/// Remaining slots padded with sentinel (weight=0.0, index=-1).
///
///   in_weights:     [num_tokens, topk], FP32 (from top-K gating)
///   in_indices:     [num_tokens, topk], INT32 (from top-K gating)
///   out_weights:    [num_tokens, topk], FP32 (sorted descending, padded)
///   out_indices:    [num_tokens, topk], INT32 (sorted descending, padded)
///   expert_counts:  [num_tokens], INT32 (number of kept experts per token)
///
void launch_adaptive_gating(float* out_weights, int32_t* out_indices,
                             int32_t* expert_counts,
                             const float* in_weights,
                             const int32_t* in_indices,
                             const AdaptiveGatingParams& params,
                             void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

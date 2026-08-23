// Top-K gating CUDA kernels for LayerStoRm MoE routing.
// Adapted from TRT-LLM RoutingKernelTopK.cuh and vLLM grouped_topk_kernels.cu
// (both Apache-2.0 — see THIRD_PARTY_NOTICES.md).
//
// Two kernel variants:
//   Grouped (n_group > 1) — V3.2 two-level routing (groups then experts)
//   Simple  (n_group = 1) — GLM-5 / K2.5 / V4 flat top-K routing
//
// Scoring functions: sigmoid (V3.2, GLM-5, K2.5), sqrtsoftplus (V4).
// Score correction bias applied for expert selection only; routing weights
// use unbiased scores.

#pragma once

#include <cstdint>

namespace layerstorm::compute {

enum class ScoringFunc : uint8_t { kSigmoid = 0, kSqrtSoftplus = 1 };

/// Parameters for top-K gating kernel launch.
struct TopkGatingParams {
    int num_tokens;
    int num_experts;
    int topk;                       // experts selected per token
    int n_group;                    // expert groups (1 = simple, >1 = grouped)
    int topk_group;                 // groups to select in phase 1 (grouped only)
    float routed_scaling_factor;    // weight scaling (V3.2: 2.5, K2.5: 2.827)
    bool renormalize;               // norm_topk_prob
    ScoringFunc scoring_func = ScoringFunc::kSigmoid;
};

/// Top-K gating with configurable scoring and optional grouped routing.
/// Auto-dispatches grouped vs simple kernel based on n_group.
///
///   logits:       [num_tokens, num_experts], FP32 (router linear output)
///   bias:         [num_experts] FP32 (e_score_correction_bias), or nullptr
///   topk_weights: [num_tokens, topk], FP32 (output routing weights)
///   topk_indices: [num_tokens, topk], INT32 (output expert indices)
///
/// Bias affects expert selection but NOT routing weights (per DeepSeek spec).
/// When renormalize=true, weights are scaled so sum == routed_scaling_factor.
void launch_topk_gating(float* topk_weights, int32_t* topk_indices,
                         const float* logits, const float* bias,
                         const TopkGatingParams& params,
                         void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

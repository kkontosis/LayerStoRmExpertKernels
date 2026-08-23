// Fused SwiGLU activation kernel for MoE expert FFN.
// Adapted from vLLM csrc/activation_kernels.cu silu_and_mul_kernel (Apache-2.0).
//
// Computes: output[i,j] = SiLU(gate[i,j]) * up[i,j]
//   where gate = input[i, 0:d], up = input[i, d:2d]
//   SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Optional clamping (V4 swiglu_limit > 0), llama.cpp DEEPSEEK4 semantics
// (llama-graph.cpp LLM_ARCH_DEEPSEEK4 branches, ggml_swiglu_split; MIT
// License, Copyright (c) 2023-2026 The ggml authors — see
// THIRD_PARTY_NOTICES.md):
//   gate = min(gate, limit)             upper bound only, before SiLU
//   up   = clamp(up, -limit, limit)     two-sided
//
// Input:  [num_tokens, 2*d] from fused gate+up grouped GEMM output
// Output: [num_tokens, d] fed into down-projection grouped GEMM

#pragma once

#include <cstdint>

namespace layerstorm::compute {

/// Parameters for fused SwiGLU kernel launch.
struct FusedSwigluParams {
    int num_tokens;        ///< total permuted tokens (from moe_permute)
    int d;                 ///< intermediate_size (half of input columns)
    float swiglu_limit;    ///< clamp bound (V4: 10.0, V3.2: 0.0 = no clamp)
};

/// Fused SwiGLU activation: output = SiLU(gate) * up.
///
///   input:   [num_tokens, 2*d] BF16/FP16 (gate in first d cols, up in last d)
///   output:  [num_tokens, d] same dtype as input
///
/// Vectorized 128-bit loads. One block per token, 256 threads.
void launch_fused_swiglu(
    void* output,             ///< [num_tokens, d]
    const void* input,        ///< [num_tokens, 2*d]
    const FusedSwigluParams& params,
    int elem_size_bytes,      ///< 2 for BF16/FP16
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

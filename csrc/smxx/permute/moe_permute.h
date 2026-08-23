// MoE token permutation and unpermutation CUDA kernels.
// Adapted from vLLM csrc/moe/permute_unpermute_kernels/ (Apache-2.0).
//
// Permute: sort tokens by expert ID (CUB radix sort), expand rows for
// multi-expert routing. Produces expert-grouped layout for grouped GEMM.
//
// Unpermute: K-way weighted reduction of expert outputs back to original
// token ordering with routing weight scaling.

#pragma once

#include <cstddef>
#include <cstdint>

namespace layerstorm::compute {

/// Query workspace bytes for MoE permute operations.
///
/// Workspace includes: CUB radix sort temp storage, sorted expert/row
/// arrays, and source-row index array.
///
///   max_tokens:  maximum tokens per batch
///   max_topk:    maximum experts selected per token (typically 8)
///   num_experts: total number of experts (e.g. 256)
size_t query_moe_permute_workspace_size(int max_tokens, int max_topk,
                                         int num_experts);

/// MoE permute: sort tokens by expert, expand rows for multi-expert routing.
///
/// Input (from gating):
///   hidden_states:  [num_tokens, hidden_dim] input activations
///   topk_indices:   [num_tokens, topk] INT32 expert IDs per token
///
/// Output:
///   permuted_input:  [num_tokens * topk, hidden_dim] tokens grouped by expert
///   expert_offsets:   [num_experts + 1] INT32 cumulative token count per expert
///   src_to_dest_map:  [num_tokens * topk] INT32 reverse permutation for unpermute
///   permuted_idx:     [num_tokens * topk] INT32 forward permutation index
///
/// Sentinel handling: topk_indices == -1 (from adaptive gating) are remapped
/// to num_experts and sorted to the end. expert_offsets covers [0..num_experts].
void launch_moe_permute(
    void* permuted_input,         ///< [num_tokens*topk, hidden_dim] output
    int32_t* expert_offsets,      ///< [num_experts + 1] output
    int32_t* src_to_dest_map,     ///< [num_tokens * topk] output (reverse map)
    int32_t* permuted_idx,        ///< [num_tokens * topk] output (forward map)
    const void* hidden_states,    ///< [num_tokens, hidden_dim] input
    const int32_t* topk_indices,  ///< [num_tokens, topk] from gating
    int num_tokens,
    int topk,
    int hidden_dim,
    int num_experts,
    int elem_size_bytes,          ///< sizeof element: 2 for BF16/FP16
    void* workspace,              ///< from query_moe_permute_workspace_size
    void* stream /*cudaStream_t*/);

/// MoE unpermute: reduce expert outputs back to original token order.
///
/// Performs weighted sum across K expert outputs per token:
///   output[i] = sum_k(expert_output[dest_map[i*topk+k]] * weight[i*topk+k])
///
///   permuted_output:  [num_tokens*topk, hidden_dim] expert FFN outputs
///   topk_weights:     [num_tokens, topk] FP32 routing weights
///   src_to_dest_map:  [num_tokens * topk] INT32 from moe_permute
///   output:           [num_tokens, hidden_dim] final reduced output
///
/// fp32_output (DET-REDUCE Phase 1b, canonical placement-INVARIANT EP combine):
/// when true, `output` is treated as an FP32 buffer [num_tokens, topk,
/// hidden_dim] and each of the K expert contributions is written to its OWN slot
/// (c_k = weight_k * expert_out_k, fp32; non-resident slots write 0) — NO
/// cross-slot sum. The expert outputs (permuted_output) remain bf16. A cross-GPU
/// fp32 allreduce then GATHERS the slots (each filled by exactly one GPU), and
/// launch_moe_combine_reduce_slots_fp32_to_bf16 sums the K slots in a FIXED
/// order and rounds to bf16 once — making the per-token combine BIT-identical
/// regardless of which GPU holds each expert. Default false keeps the legacy
/// bf16-in/bf16-out summed path (byte-identical). (A simpler fp32 partial-SUM
/// per GPU was tried and is NOT placement-invariant — fp32 is non-associative
/// across a changing partition — hence the per-slot canonical form.)
void launch_moe_unpermute(
    void* output,                    ///< [num_tokens, hidden_dim] (bf16) OR [num_tokens, topk, hidden_dim] (fp32 per-slot)
    const void* permuted_output,     ///< [num_tokens*topk, hidden_dim]
    const float* topk_weights,       ///< [num_tokens, topk]
    const int32_t* src_to_dest_map,  ///< [num_tokens * topk]
    int num_tokens,
    int topk,
    int hidden_dim,
    int elem_size_bytes,             ///< sizeof element: 2 for BF16/FP16
    void* stream /*cudaStream_t*/,
    bool fp32_output = false);

/// MoE unpermute → BF16 PER-SLOT (DET-REDUCE Phase 1b, canonical placement-INVARIANT
/// EP combine — BF16-PAYLOAD variant). Identical to launch_moe_unpermute(fp32_output=
/// true) except each per-slot contribution c_k = weight_k * expert_out_k is rounded
/// to BF16 once and written to a BF16 buffer [num_tokens, topk, hidden_dim] — half the
/// gather bytes of the fp32 per-slot path. The cross-GPU gather is a bf16 SUM-allreduce
/// (exact: each slot filled by exactly one GPU, 0+x=x), then
/// launch_moe_combine_reduce_slots_bf16_to_bf16 sums the K slots in FIXED order and
/// rounds to bf16 once. Placement-invariant by the fixed slot ORDER (not the payload
/// dtype); matches the vLLM/llama.cpp convention (fp32 math, 16-bit payload).
void launch_moe_unpermute_bf16_perslot(
    void* output,                    ///< [num_tokens, topk, hidden_dim] BF16 per-slot
    const void* permuted_output,     ///< [num_tokens*topk, hidden_dim] BF16
    const float* topk_weights,       ///< [num_tokens, topk]
    const int32_t* src_to_dest_map,  ///< [num_tokens * topk]
    int num_tokens,
    int topk,
    int hidden_dim,
    int elem_size_bytes,             ///< sizeof element: 2 for BF16/FP16
    void* stream /*cudaStream_t*/);

/// Fixed-order K-slot reduce → BF16 for the canonical placement-invariant EP
/// combine (DET-REDUCE Phase 1b). Sums the topk per-slot fp32 contributions in
/// canonical slot order 0..topk-1 and rounds to bf16 once (round-to-nearest).
///   output_bf16:        [num_tokens, hidden_dim] BF16 destination
///   input_perslot_fp32: [num_tokens, topk, hidden_dim] FP32 (allreduced slots)
void launch_moe_combine_reduce_slots_fp32_to_bf16(
    void* output_bf16,
    const void* input_perslot_fp32,
    int num_tokens,
    int topk,
    int hidden_dim,
    void* stream /*cudaStream_t*/);

/// BF16-PAYLOAD variant of launch_moe_combine_reduce_slots_fp32_to_bf16: the
/// gathered per-slot input is BF16 [num_tokens, topk, hidden_dim]; each slot is
/// widened to fp32 and accumulated in FIXED slot order 0..topk-1 (fp32 math),
/// then rounded to bf16 ONCE → output_bf16 [num_tokens, hidden_dim].
void launch_moe_combine_reduce_slots_bf16_to_bf16(
    void* output_bf16,
    const void* input_perslot_bf16,
    int num_tokens,
    int topk,
    int hidden_dim,
    void* stream /*cudaStream_t*/);

}  // namespace layerstorm::compute

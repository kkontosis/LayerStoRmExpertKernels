// Pybind11 Python wrappers for SM120 Expert/MoE CUDA kernels.
// #included from bindings.cu — not compiled standalone.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <string>
#include <cstdint>

// All kernel headers already visible via bindings.cu includes.
// Namespace alias for convenience.
namespace lc = layerstorm::compute;

// ── Helpers ──────────────────────────────────────────────────────────────────

static void* current_stream() {
    return static_cast<void*>(at::cuda::getCurrentCUDAStream().stream());
}

// ── SwiGLU ───────────────────────────────────────────────────────────────────

torch::Tensor fused_swiglu(torch::Tensor input, double swiglu_limit) {
    TORCH_CHECK(input.is_cuda(), "fused_swiglu: input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "fused_swiglu: input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "fused_swiglu: input must be 2D [num_tokens, 2*d]");
    TORCH_CHECK(input.size(1) % 2 == 0,
                "fused_swiglu: last dim must be even (gate|up interleaved)");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 ||
                input.scalar_type() == at::kHalf,
                "fused_swiglu: input must be BF16 or FP16");
    TORCH_CHECK(swiglu_limit >= 0.0,
                "fused_swiglu: swiglu_limit must be >= 0 (0 = no clamp)");

    int num_tokens = input.size(0);
    int d = input.size(1) / 2;

    auto output = torch::empty({num_tokens, d}, input.options());

    lc::FusedSwigluParams params;
    params.num_tokens = num_tokens;
    params.d = d;
    params.swiglu_limit = static_cast<float>(swiglu_limit);

    lc::launch_fused_swiglu(output.data_ptr(), input.data_ptr(),
                            params, input.element_size(), current_stream());
    return output;
}

// ── MoE Permute ──────────────────────────────────────────────────────────────

std::vector<torch::Tensor> moe_permute(
    torch::Tensor hidden_states,
    torch::Tensor topk_indices,
    int64_t num_experts) {

    TORCH_CHECK(hidden_states.is_cuda(), "moe_permute: hidden_states must be on CUDA");
    TORCH_CHECK(topk_indices.is_cuda(), "moe_permute: topk_indices must be on CUDA");
    TORCH_CHECK(hidden_states.is_contiguous(), "moe_permute: hidden_states must be contiguous");
    TORCH_CHECK(topk_indices.is_contiguous(), "moe_permute: topk_indices must be contiguous");
    TORCH_CHECK(hidden_states.dim() == 2, "moe_permute: hidden_states must be 2D");
    TORCH_CHECK(topk_indices.dim() == 2, "moe_permute: topk_indices must be 2D [num_tokens, topk]");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt,
                "moe_permute: topk_indices must be int32");

    int num_tokens = hidden_states.size(0);
    int hidden_dim = hidden_states.size(1);
    int topk = topk_indices.size(1);
    int expanded = num_tokens * topk;

    // Allocate outputs
    auto permuted_input = torch::empty({expanded, hidden_dim}, hidden_states.options());
    auto expert_offsets = torch::empty({static_cast<int64_t>(num_experts) + 1},
                                        topk_indices.options());  // int32
    auto src_to_dest_map = torch::empty({expanded}, topk_indices.options());
    auto permuted_idx = torch::empty({expanded}, topk_indices.options());

    // Allocate workspace
    size_t ws_bytes = lc::query_moe_permute_workspace_size(
        num_tokens, topk, static_cast<int>(num_experts));
    auto workspace = torch::empty({static_cast<int64_t>(ws_bytes)},
                                   hidden_states.options().dtype(at::kByte));

    lc::launch_moe_permute(
        permuted_input.data_ptr(),
        expert_offsets.data_ptr<int32_t>(),
        src_to_dest_map.data_ptr<int32_t>(),
        permuted_idx.data_ptr<int32_t>(),
        hidden_states.data_ptr(),
        topk_indices.data_ptr<int32_t>(),
        num_tokens, topk, hidden_dim,
        static_cast<int>(num_experts),
        hidden_states.element_size(),
        workspace.data_ptr(),
        current_stream());

    return {permuted_input, expert_offsets, src_to_dest_map, permuted_idx};
}

// ── MoE Unpermute ────────────────────────────────────────────────────────────

torch::Tensor moe_unpermute(
    torch::Tensor permuted_output,
    torch::Tensor topk_weights,
    torch::Tensor src_to_dest_map,
    int64_t num_tokens,
    int64_t topk) {

    TORCH_CHECK(permuted_output.is_cuda(), "moe_unpermute: permuted_output must be on CUDA");
    TORCH_CHECK(topk_weights.is_cuda(), "moe_unpermute: topk_weights must be on CUDA");
    TORCH_CHECK(src_to_dest_map.is_cuda(), "moe_unpermute: src_to_dest_map must be on CUDA");
    TORCH_CHECK(permuted_output.is_contiguous(), "moe_unpermute: permuted_output must be contiguous");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
                "moe_unpermute: topk_weights must be float32");
    TORCH_CHECK(src_to_dest_map.scalar_type() == at::kInt,
                "moe_unpermute: src_to_dest_map must be int32");

    int hidden_dim = permuted_output.size(1);
    auto output = torch::empty({num_tokens, hidden_dim}, permuted_output.options());

    lc::launch_moe_unpermute(
        output.data_ptr(),
        permuted_output.data_ptr(),
        topk_weights.data_ptr<float>(),
        src_to_dest_map.data_ptr<int32_t>(),
        static_cast<int>(num_tokens),
        static_cast<int>(topk),
        hidden_dim,
        permuted_output.element_size(),
        current_stream());

    return output;
}

// ── MoE Unpermute → FP32 PER-SLOT (DET-REDUCE Phase 1b canonical EP combine) ──

torch::Tensor moe_unpermute_fp32(
    torch::Tensor permuted_output,
    torch::Tensor topk_weights,
    torch::Tensor src_to_dest_map,
    int64_t num_tokens,
    int64_t topk) {

    TORCH_CHECK(permuted_output.is_cuda(), "moe_unpermute_fp32: permuted_output must be on CUDA");
    TORCH_CHECK(topk_weights.is_cuda(), "moe_unpermute_fp32: topk_weights must be on CUDA");
    TORCH_CHECK(src_to_dest_map.is_cuda(), "moe_unpermute_fp32: src_to_dest_map must be on CUDA");
    TORCH_CHECK(permuted_output.is_contiguous(), "moe_unpermute_fp32: permuted_output must be contiguous");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
                "moe_unpermute_fp32: topk_weights must be float32");
    TORCH_CHECK(src_to_dest_map.scalar_type() == at::kInt,
                "moe_unpermute_fp32: src_to_dest_map must be int32");
    TORCH_CHECK(permuted_output.element_size() == 2,
                "moe_unpermute_fp32: permuted_output must be bf16/fp16 (2 bytes)");

    int hidden_dim = permuted_output.size(1);
    // Per-slot fp32 output: [num_tokens, topk, hidden_dim].
    auto output = torch::empty(
        {num_tokens, topk, hidden_dim},
        permuted_output.options().dtype(at::kFloat));

    lc::launch_moe_unpermute(
        output.data_ptr(),
        permuted_output.data_ptr(),
        topk_weights.data_ptr<float>(),
        src_to_dest_map.data_ptr<int32_t>(),
        static_cast<int>(num_tokens),
        static_cast<int>(topk),
        hidden_dim,
        permuted_output.element_size(),
        current_stream(),
        /*fp32_output=*/true);

    return output;
}

// Fixed-order K-slot reduce → BF16 used after the fp32 cross-GPU EP gather.
// input_perslot_fp32: [num_tokens, topk, hidden_dim] fp32.
torch::Tensor moe_combine_reduce_slots_fp32_to_bf16(torch::Tensor input_perslot_fp32) {
    TORCH_CHECK(input_perslot_fp32.is_cuda(), "moe_combine_reduce: input must be on CUDA");
    TORCH_CHECK(input_perslot_fp32.is_contiguous(), "moe_combine_reduce: input must be contiguous");
    TORCH_CHECK(input_perslot_fp32.scalar_type() == at::kFloat,
                "moe_combine_reduce: input must be float32");
    TORCH_CHECK(input_perslot_fp32.dim() == 3,
                "moe_combine_reduce: input must be [num_tokens, topk, hidden]");
    int num_tokens = input_perslot_fp32.size(0);
    int topk = input_perslot_fp32.size(1);
    int hidden_dim = input_perslot_fp32.size(2);
    auto output = torch::empty(
        {num_tokens, hidden_dim},
        input_perslot_fp32.options().dtype(at::kBFloat16));
    lc::launch_moe_combine_reduce_slots_fp32_to_bf16(
        output.data_ptr(), input_perslot_fp32.data_ptr(),
        num_tokens, topk, hidden_dim, current_stream());
    return output;
}

// ── MoE Unpermute → BF16 PER-SLOT (DET-REDUCE Phase 1b canonical, BF16 payload) ──

torch::Tensor moe_unpermute_bf16_perslot(
    torch::Tensor permuted_output,
    torch::Tensor topk_weights,
    torch::Tensor src_to_dest_map,
    int64_t num_tokens,
    int64_t topk) {

    TORCH_CHECK(permuted_output.is_cuda(), "moe_unpermute_bf16_perslot: permuted_output must be on CUDA");
    TORCH_CHECK(topk_weights.is_cuda(), "moe_unpermute_bf16_perslot: topk_weights must be on CUDA");
    TORCH_CHECK(src_to_dest_map.is_cuda(), "moe_unpermute_bf16_perslot: src_to_dest_map must be on CUDA");
    TORCH_CHECK(permuted_output.is_contiguous(), "moe_unpermute_bf16_perslot: permuted_output must be contiguous");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
                "moe_unpermute_bf16_perslot: topk_weights must be float32");
    TORCH_CHECK(src_to_dest_map.scalar_type() == at::kInt,
                "moe_unpermute_bf16_perslot: src_to_dest_map must be int32");
    TORCH_CHECK(permuted_output.element_size() == 2,
                "moe_unpermute_bf16_perslot: permuted_output must be bf16/fp16 (2 bytes)");

    int hidden_dim = permuted_output.size(1);
    // Per-slot bf16 output: [num_tokens, topk, hidden_dim].
    auto output = torch::empty(
        {num_tokens, topk, hidden_dim}, permuted_output.options());

    lc::launch_moe_unpermute_bf16_perslot(
        output.data_ptr(),
        permuted_output.data_ptr(),
        topk_weights.data_ptr<float>(),
        src_to_dest_map.data_ptr<int32_t>(),
        static_cast<int>(num_tokens),
        static_cast<int>(topk),
        hidden_dim,
        permuted_output.element_size(),
        current_stream());

    return output;
}

// BF16-payload fixed-order K-slot reduce used after the bf16 cross-GPU EP gather.
// input_perslot_bf16: [num_tokens, topk, hidden_dim] bf16.
torch::Tensor moe_combine_reduce_slots_bf16_to_bf16(torch::Tensor input_perslot_bf16) {
    TORCH_CHECK(input_perslot_bf16.is_cuda(), "moe_combine_reduce_bf16: input must be on CUDA");
    TORCH_CHECK(input_perslot_bf16.is_contiguous(), "moe_combine_reduce_bf16: input must be contiguous");
    TORCH_CHECK(input_perslot_bf16.element_size() == 2,
                "moe_combine_reduce_bf16: input must be bf16/fp16 (2 bytes)");
    TORCH_CHECK(input_perslot_bf16.dim() == 3,
                "moe_combine_reduce_bf16: input must be [num_tokens, topk, hidden]");
    int num_tokens = input_perslot_bf16.size(0);
    int topk = input_perslot_bf16.size(1);
    int hidden_dim = input_perslot_bf16.size(2);
    auto output = torch::empty(
        {num_tokens, hidden_dim}, input_perslot_bf16.options());
    lc::launch_moe_combine_reduce_slots_bf16_to_bf16(
        output.data_ptr(), input_perslot_bf16.data_ptr(),
        num_tokens, topk, hidden_dim, current_stream());
    return output;
}

// ── Adaptive Gating ──────────────────────────────────────────────────────────

std::vector<torch::Tensor> adaptive_gating(
    torch::Tensor in_weights,
    torch::Tensor in_indices,
    double threshold) {

    TORCH_CHECK(in_weights.is_cuda(), "adaptive_gating: in_weights must be on CUDA");
    TORCH_CHECK(in_indices.is_cuda(), "adaptive_gating: in_indices must be on CUDA");
    TORCH_CHECK(in_weights.is_contiguous() && in_indices.is_contiguous(),
                "adaptive_gating: inputs must be contiguous");
    TORCH_CHECK(in_weights.scalar_type() == at::kFloat,
                "adaptive_gating: in_weights must be float32");
    TORCH_CHECK(in_indices.scalar_type() == at::kInt,
                "adaptive_gating: in_indices must be int32");
    TORCH_CHECK(in_weights.dim() == 2, "adaptive_gating: in_weights must be 2D");

    int num_tokens = in_weights.size(0);
    int topk = in_weights.size(1);

    auto out_weights = torch::empty_like(in_weights);
    auto out_indices = torch::empty_like(in_indices);
    auto expert_counts = torch::empty({num_tokens},
                                       in_indices.options());  // int32

    lc::AdaptiveGatingParams params;
    params.num_tokens = num_tokens;
    params.topk = topk;
    params.threshold = static_cast<float>(threshold);

    lc::launch_adaptive_gating(
        out_weights.data_ptr<float>(),
        out_indices.data_ptr<int32_t>(),
        expert_counts.data_ptr<int32_t>(),
        in_weights.data_ptr<float>(),
        in_indices.data_ptr<int32_t>(),
        params, current_stream());

    return {out_weights, out_indices, expert_counts};
}

// ── Top-K Gating ─────────────────────────────────────────────────────────────

std::vector<torch::Tensor> topk_gating(
    torch::Tensor logits,
    c10::optional<torch::Tensor> bias,
    int64_t topk,
    int64_t n_group,
    int64_t topk_group,
    double routed_scaling_factor,
    bool renormalize,
    std::string scoring_func) {

    TORCH_CHECK(logits.is_cuda(), "topk_gating: logits must be on CUDA");
    TORCH_CHECK(logits.is_contiguous(), "topk_gating: logits must be contiguous");
    TORCH_CHECK(logits.scalar_type() == at::kFloat,
                "topk_gating: logits must be float32");
    TORCH_CHECK(logits.dim() == 2, "topk_gating: logits must be 2D [num_tokens, num_experts]");

    lc::ScoringFunc sf;
    if (scoring_func == "sigmoid") {
        sf = lc::ScoringFunc::kSigmoid;
    } else if (scoring_func == "sqrtsoftplus") {
        sf = lc::ScoringFunc::kSqrtSoftplus;
    } else {
        TORCH_CHECK(false, "topk_gating: scoring_func must be 'sigmoid' or 'sqrtsoftplus', got '",
                    scoring_func, "'");
    }

    int num_tokens = logits.size(0);
    int num_experts = logits.size(1);

    auto topk_weights = torch::empty({num_tokens, static_cast<int64_t>(topk)},
                                      logits.options());
    auto topk_indices = torch::empty({num_tokens, static_cast<int64_t>(topk)},
                                      logits.options().dtype(at::kInt));

    float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "topk_gating: bias must be on CUDA");
        TORCH_CHECK(bias->is_contiguous(), "topk_gating: bias must be contiguous");
        TORCH_CHECK(bias->scalar_type() == at::kFloat,
                    "topk_gating: bias must be float32");
        bias_ptr = bias->data_ptr<float>();
    }

    lc::TopkGatingParams params;
    params.num_tokens = num_tokens;
    params.num_experts = num_experts;
    params.topk = static_cast<int>(topk);
    params.n_group = static_cast<int>(n_group);
    params.topk_group = static_cast<int>(topk_group);
    params.routed_scaling_factor = static_cast<float>(routed_scaling_factor);
    params.renormalize = renormalize;
    params.scoring_func = sf;

    lc::launch_topk_gating(
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        bias_ptr,
        params, current_stream());

    return {topk_weights, topk_indices};
}

// ── Module definition ────────────────────────────────────────────────────────
// GEMM kernels (nvfp4, fp8, grouped) and dynamic_fp8_quant are available
// via sm120_gemm_kernels (LayerStoRmGemmKernels). This module exposes only
// expert/MoE-specific kernels.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SM120 Expert/MoE CUDA kernels";

    // Arch-generic kernels
    m.def("fused_swiglu", &fused_swiglu,
          py::arg("input"), py::arg("swiglu_limit") = 0.0,
          "Fused SwiGLU: SiLU(gate) * up with optional clamp. Input [N, 2*d] -> output [N, d].");

    m.def("moe_permute", &moe_permute,
          py::arg("hidden_states"), py::arg("topk_indices"),
          py::arg("num_experts"),
          "MoE permute: sort tokens by expert. Returns (permuted, offsets, s2d_map, perm_idx).");

    m.def("moe_unpermute", &moe_unpermute,
          py::arg("permuted_output"), py::arg("topk_weights"),
          py::arg("src_to_dest_map"), py::arg("num_tokens"), py::arg("topk"),
          "MoE unpermute: weighted reduction back to original order.");

    m.def("moe_unpermute_fp32", &moe_unpermute_fp32,
          py::arg("permuted_output"), py::arg("topk_weights"),
          py::arg("src_to_dest_map"), py::arg("num_tokens"), py::arg("topk"),
          "MoE unpermute (DET-REDUCE Phase 1b): per-slot fp32 contributions "
          "[num_tokens, topk, hidden] (no cross-slot sum).");

    m.def("moe_combine_reduce_slots_fp32_to_bf16", &moe_combine_reduce_slots_fp32_to_bf16,
          py::arg("input_perslot_fp32"),
          "DET-REDUCE Phase 1b: fixed-order K-slot reduce of [num_tokens, topk, "
          "hidden] fp32 → [num_tokens, hidden] bf16 (placement-invariant).");

    m.def("moe_unpermute_bf16_perslot", &moe_unpermute_bf16_perslot,
          py::arg("permuted_output"), py::arg("topk_weights"),
          py::arg("src_to_dest_map"), py::arg("num_tokens"), py::arg("topk"),
          "MoE unpermute (DET-REDUCE Phase 1b, BF16 payload): per-slot bf16 "
          "contributions [num_tokens, topk, hidden] (no cross-slot sum).");

    m.def("moe_combine_reduce_slots_bf16_to_bf16", &moe_combine_reduce_slots_bf16_to_bf16,
          py::arg("input_perslot_bf16"),
          "DET-REDUCE Phase 1b (BF16 payload): fixed-order K-slot reduce of "
          "[num_tokens, topk, hidden] bf16 → [num_tokens, hidden] bf16 (fp32 accumulate).");

    m.def("adaptive_gating", &adaptive_gating,
          py::arg("in_weights"), py::arg("in_indices"), py::arg("threshold"),
          "Adaptive gating: threshold-based expert pruning. Returns (weights, indices, counts).");

    // SM120 gating
    m.def("topk_gating", &topk_gating,
          py::arg("logits"), py::arg("bias"),
          py::arg("topk"), py::arg("n_group"), py::arg("topk_group"),
          py::arg("routed_scaling_factor"), py::arg("renormalize"),
          py::arg("scoring_func") = "sigmoid",
          "Top-K gating with configurable scoring. Returns (weights, indices).");
}

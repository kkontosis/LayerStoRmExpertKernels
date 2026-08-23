# USAGE.md — SM120 Expert Kernel Library

## What This Is

CUDA kernel library for MoE (Mixture of Experts) inference on SM120 GPUs (RTX 5090/5080). Provides the complete expert forward pipeline: grouped GEMM with NVFP4/FP8 quantized weights, SwiGLU activation, and expert routing permute/unpermute.

All grouped GEMM kernels support **NVFP4** (primary, 4-bit weights) and **FP8** (secondary, 8-bit weights) quantization formats.

## Weight Formats

### NVFP4 (Primary — Routed Experts)

FP4 E2M1 with UE8M0 block scales, group_size=16:
- 0.5625 bytes/element
- 2 FP4 values packed per byte
- 1 UE8M0 scale per 16 elements

### FP8 (Secondary — Shared Experts)

FP8 E4M3 or E5M2:
- 1.0 bytes/element, no additional scales

## Kernel Inventory

| Kernel | File | Precision | Purpose |
|--------|------|-----------|---------|
| **nvfp4_gemm** | `sm120/gemm/nvfp4/nvfp4_gemm.cu` | NVFP4, BF16 act | Single-expert NVFP4 GEMM |
| **nvfp4_grouped_gemm** | `sm120/gemm/nvfp4/nvfp4_grouped_gemm.cu` | NVFP4, BF16 act | Multi-expert NVFP4 grouped GEMM |
| **fp8_gemm** | `sm120/gemm/fp8/fp8_gemm.cu` | FP8 E4M3, blockwise scales | Single-expert FP8 GEMM (M≥128) |
| **fp8_grouped_gemm** | `sm120/gemm/fp8/fp8_grouped_gemm.cu` | FP8 E4M3, blockwise scales | Multi-expert FP8 dispatch loop |
| **topk_gating** | `sm120/gating/topk_gating.cu` | FP32 | Warp-level top-K (sigmoid/sqrtsoftplus scoring) |
| **fused_swiglu** | `smxx/activation/fused_swiglu.cu` | BF16 | SiLU(gate) * up with optional clamp |
| **moe_permute** | `smxx/permute/moe_permute.cu` | BF16 | CUB radix sort token scatter |
| **moe_unpermute** | `smxx/permute/moe_permute.cu` | BF16 | K-way weighted gather |
| **adaptive_gating** | `smxx/gating/adaptive_gating.cu` | FP32 | Threshold-based expert pruning |
| **dynamic_fp8_quant** | `smxx/quant/dynamic_fp8_quant.cu` | BF16→FP8 | Per-block FP8 quantization |

## Expert Forward Flow

```
Input: tokens [B, hidden_dim] BF16 + expert assignments from gating
  │
  ├─ expert_permute(tokens, expert_ids, expert_offsets)
  │     → permuted_tokens [total_tokens, hidden_dim] (expert-grouped)
  │
  ├─ grouped_gemm_nvfp4(W_gate, permuted_tokens) → gate_out [total_tokens, inter_dim]
  ├─ grouped_gemm_nvfp4(W_up,   permuted_tokens) → up_out   [total_tokens, inter_dim]
  │
  ├─ swiglu(gate_out, up_out) → activated [total_tokens, inter_dim]
  │
  ├─ grouped_gemm_nvfp4(W_down, activated) → expert_out [total_tokens, hidden_dim]
  │
  └─ expert_unpermute(expert_out, expert_ids, gating_weights)
        → output [B, hidden_dim] (original order, weighted sum)
```

## C++ API

Include headers from `csrc/`. All functions in `namespace layerstorm::compute`.

```cpp
#include "smxx/activation/fused_swiglu.h"
#include "smxx/permute/moe_permute.h"
#include "sm120/gemm/grouped_gemm.h"

using namespace layerstorm::compute;

// SwiGLU: input [N, 2*d] → output [N, d]
// swiglu_limit: 0.0 = no clamp (V3.2), 10.0 = V4 clamp
FusedSwigluParams params{.num_tokens = N, .d = d, .swiglu_limit = 0.0f};
launch_fused_swiglu(output, input, params, elem_size_bytes, stream);

// MoE Permute
size_t ws = query_moe_permute_workspace_size(max_tokens, topk, num_experts);
launch_moe_permute(permuted, offsets, s2d_map, perm_idx,
                   hidden_states, topk_indices, ...);
launch_moe_unpermute(output, permuted, weights, s2d_map, ...);

// Top-K Gating
// scoring_func: kSigmoid (V3.2), kSqrtSoftplus (V4)
TopkGatingParams gp{.num_tokens=N, .num_experts=256, .topk=8,
                     .scoring_func = ScoringFunc::kSigmoid, ...};
launch_topk_gating(weights, indices, logits, bias, gp, stream);

// FP8 GEMM
Fp8GemmParams fp{.M=M, .N=N, .K=K, .A=a, .B=b, .D=d, ...};
launch_fp8_gemm(fp, workspace, stream);
```

## Python API

```python
import sm120_expert_kernels as EK

# SwiGLU: input [N, 2*d] BF16 → output [N, d]
output = EK.fused_swiglu(input)                      # V3.2 (no clamp)
output = EK.fused_swiglu(input, swiglu_limit=10.0)   # V4 (gate clamp + SiLU cap)

# MoE Permute/Unpermute
permuted, offsets, s2d_map, perm_idx = EK.moe_permute(tokens, topk_indices, num_experts)
output = EK.moe_unpermute(permuted, topk_weights, s2d_map, num_tokens, topk)

# Top-K Gating (grouped or simple)
weights, indices = EK.topk_gating(logits, bias, topk=8, n_group=8,
                                   topk_group=4, routed_scaling_factor=2.5,
                                   renormalize=True)  # V3.2 sigmoid (default)
weights, indices = EK.topk_gating(logits, bias, topk=6, n_group=1,
                                   topk_group=1, routed_scaling_factor=2.5,
                                   renormalize=True,
                                   scoring_func="sqrtsoftplus")  # V4

# Adaptive Gating
out_w, out_i, counts = EK.adaptive_gating(in_weights, in_indices, threshold=0.9)

# Dynamic FP8 Quantization: BF16 → FP8 E4M3
output_fp8, scales = EK.dynamic_fp8_quant(input_bf16)

# FP8 GEMM (SM120, M >= 128)
D = EK.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B, M, N, K, "bf16")
EK.fp8_grouped_gemm(A, B, D, scale_A, scale_B, offsets, sizes, N, K, "bf16")

# NVFP4 GEMM (SM120)
D = EK.nvfp4_gemm(A, B, scale_A, scale_B, M, N, K, alpha=1.0, output_dtype="bf16")
EK.nvfp4_grouped_gemm(A, B, D, sA, sB, alphas, offsets, sf_offsets, sizes, N, K, "bf16")

# Workspace queries
ws = EK.query_fp8_gemm_workspace_size(M, N, K, "bf16")
ws = EK.query_nvfp4_gemm_workspace_size(M, N, K, "bf16")
```

## Model Dimensions

| Parameter | V3.2 (DeepSeek) | V4 (DeepSeek) | MODEL1 (Kimi) |
|-----------|-----------------|---------------|---------------|
| hidden_dim | 7168 | 7168 | 6144 |
| intermediate_dim (per expert) | 2048 | 2048 | 1536 |
| n_routed_experts | 256 | 256 | 128 |
| n_shared_experts | 1 | 1 | 1 |
| top_k (experts per token) | 8 | 6 | 8 |
| n_group | 8 | 1 (flat) | 1 |
| scoring_func | sigmoid | sqrtsoftplus | sigmoid |
| swiglu_limit | 0 (none) | 10.0 | 0 (none) |
| Routed expert quant | NVFP4 | NVFP4 | NVFP4 |
| Shared expert quant | FP8 | FP8 | FP8 |

### Per-Expert Weight Sizes

| Weight | V3.2 (NVFP4) | V3.2 (FP8) | MODEL1 (NVFP4) | MODEL1 (FP8) |
|--------|-------------|-----------|---------------|-------------|
| W_gate | 7168×2048 × 0.5625 = 8.25 MB | 14.0 MB | 6144×1536 × 0.5625 = 5.31 MB | 9.0 MB |
| W_up | 8.25 MB | 14.0 MB | 5.31 MB | 9.0 MB |
| W_down | 2048×7168 × 0.5625 = 8.25 MB | 14.0 MB | 1536×6144 × 0.5625 = 5.31 MB | 9.0 MB |
| **Total/expert** | **24.75 MB** | **42.0 MB** | **15.94 MB** | **27.0 MB** |

## Build

```bash
pip install -e . --no-build-isolation    # Python extension
```

Requires: CUDA 12.8+ (SM120), CUTLASS 3.x, PyTorch 2.x.

For C++ integration: include `csrc/` headers, compile `.cu` files with `-arch=sm_120 -std=c++17`. See `setup.py` for the full source list and compiler flags.

## License

Licensed under the Apache License 2.0 — see `LICENSE.md`. Third-party
attributions and license notices (vLLM, TensorRT-LLM, llama.cpp, FlashMLA,
CUTLASS) are collected in `THIRD_PARTY_NOTICES.md`.

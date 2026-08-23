// Fused SwiGLU activation kernel: output = SiLU(gate) * up.
//
// Adapted from vLLM csrc/activation_kernels.cu (Apache-2.0).
// Original: act_and_mul_kernel + silu_kernel.
// Stripped PyTorch/c10 dependencies: raw CUDA pointers + explicit stream.
//
// Optional clamping (swiglu_limit > 0), llama.cpp DEEPSEEK4 semantics
// (llama.cpp: MIT License, Copyright (c) 2023-2026 The ggml authors —
// see THIRD_PARTY_NOTICES.md)
// (llama-graph.cpp build_ffn/build_moe_ffn LLM_ARCH_DEEPSEEK4 branch):
//   gate = min(gate, limit)            (upper bound only)
//   up   = clamp(up, -limit, limit)    (two-sided)
//   out  = SiLU(gate) * up
// Architecture-agnostic (no SM120-specific ops). Vectorized 128-bit I/O.

#include "smxx/activation/fused_swiglu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace {

// ── BF16 SwiGLU kernel ────────────────────────────────────────────────────
// One block per token. 128-bit vectorized: 8 BF16 elements per load.
// gate = input[token, 0:d], up = input[token, d:2d]
// output[token, j] = SiLU(gate[j]) * up[j]

__global__ void fused_swiglu_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    int d, float swiglu_limit) {

    const int token = blockIdx.x;
    const __nv_bfloat16* gate = input + static_cast<int64_t>(token) * 2 * d;
    const __nv_bfloat16* up = gate + d;
    __nv_bfloat16* out = output + static_cast<int64_t>(token) * d;

    const bool do_clamp = swiglu_limit > 0.0f;

    // Vectorized path: 8 BF16 per int4 (128 bits)
    constexpr int ELEMS_PER_VEC = 8;
    const int num_vecs = d / ELEMS_PER_VEC;

    const int4* gate_vec = reinterpret_cast<const int4*>(gate);
    const int4* up_vec = reinterpret_cast<const int4*>(up);
    int4* out_vec = reinterpret_cast<int4*>(out);

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        int4 g = gate_vec[i];
        int4 u = up_vec[i];

        __nv_bfloat16* gp = reinterpret_cast<__nv_bfloat16*>(&g);
        __nv_bfloat16* up_p = reinterpret_cast<__nv_bfloat16*>(&u);

        int4 result;
        __nv_bfloat16* rp = reinterpret_cast<__nv_bfloat16*>(&result);

        #pragma unroll
        for (int j = 0; j < ELEMS_PER_VEC; j++) {
            float gf = __bfloat162float(gp[j]);
            float uf = __bfloat162float(up_p[j]);
            if (do_clamp) {
                gf = fminf(gf, swiglu_limit);
                uf = fminf(fmaxf(uf, -swiglu_limit), swiglu_limit);
            }
            float silu_g = gf * (0.5f * tanhf(0.5f * gf) + 0.5f);
            rp[j] = __float2bfloat16(silu_g * uf);
        }

        out_vec[i] = result;
    }

    // Scalar remainder
    int remainder_start = num_vecs * ELEMS_PER_VEC;
    for (int j = remainder_start + threadIdx.x; j < d; j += blockDim.x) {
        float gf = __bfloat162float(gate[j]);
        float uf = __bfloat162float(up[j]);
        if (do_clamp) {
            gf = fminf(gf, swiglu_limit);
            uf = fminf(fmaxf(uf, -swiglu_limit), swiglu_limit);
        }
        float silu_g = gf / (1.0f + expf(-gf));
        out[j] = __float2bfloat16(silu_g * uf);
    }
}

// ── FP16 SwiGLU kernel ────────────────────────────────────────────────────

__global__ void fused_swiglu_fp16(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    int d, float swiglu_limit) {

    const int token = blockIdx.x;
    const __half* gate = input + static_cast<int64_t>(token) * 2 * d;
    const __half* up = gate + d;
    __half* out = output + static_cast<int64_t>(token) * d;

    const bool do_clamp = swiglu_limit > 0.0f;

    constexpr int ELEMS_PER_VEC = 8;
    const int num_vecs = d / ELEMS_PER_VEC;

    const int4* gate_vec = reinterpret_cast<const int4*>(gate);
    const int4* up_vec = reinterpret_cast<const int4*>(up);
    int4* out_vec = reinterpret_cast<int4*>(out);

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        int4 g = gate_vec[i];
        int4 u = up_vec[i];

        __half* gp = reinterpret_cast<__half*>(&g);
        __half* up_p = reinterpret_cast<__half*>(&u);

        int4 result;
        __half* rp = reinterpret_cast<__half*>(&result);

        #pragma unroll
        for (int j = 0; j < ELEMS_PER_VEC; j++) {
            float gf = __half2float(gp[j]);
            float uf = __half2float(up_p[j]);
            if (do_clamp) {
                gf = fminf(gf, swiglu_limit);
                uf = fminf(fmaxf(uf, -swiglu_limit), swiglu_limit);
            }
            float silu_g = gf * (0.5f * tanhf(0.5f * gf) + 0.5f);
            rp[j] = __float2half(silu_g * uf);
        }

        out_vec[i] = result;
    }

    int remainder_start = num_vecs * ELEMS_PER_VEC;
    for (int j = remainder_start + threadIdx.x; j < d; j += blockDim.x) {
        float gf = __half2float(gate[j]);
        float uf = __half2float(up[j]);
        if (do_clamp) {
            gf = fminf(gf, swiglu_limit);
            uf = fminf(fmaxf(uf, -swiglu_limit), swiglu_limit);
        }
        float silu_g = gf / (1.0f + expf(-gf));
        out[j] = __float2half(silu_g * uf);
    }
}

}  // anonymous namespace

// ── Public API ─────────────────────────────────────────────────────────────

namespace layerstorm::compute {

void launch_fused_swiglu(
    void* output,
    const void* input,
    const FusedSwigluParams& params,
    int elem_size_bytes,
    void* stream) {

    if (params.num_tokens <= 0 || params.d <= 0) return;

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    int blocks = params.num_tokens;
    int threads = std::min(std::max(params.d / 8, 1), 1024);
    // Ensure at least 32 threads (one warp)
    threads = std::max(threads, 32);

    if (elem_size_bytes == 2) {
        // BF16 path (also used for FP16 since both are 2 bytes;
        // caller is responsible for matching types)
        fused_swiglu_bf16<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<__nv_bfloat16*>(output),
            static_cast<const __nv_bfloat16*>(input),
            params.d, params.swiglu_limit);
    } else {
        throw std::runtime_error(
            "fused_swiglu: unsupported elem_size_bytes=" +
            std::to_string(elem_size_bytes));
    }
}

}  // namespace layerstorm::compute

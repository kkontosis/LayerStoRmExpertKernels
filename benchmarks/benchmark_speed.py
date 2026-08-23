"""
Kernel latency benchmarking for SM120 Expert kernels.
Uses CUDA events for timing (not wall clock).

Usage:
    python benchmarks/benchmark_speed.py -v [-o results.json]

Convention (matching LayerStoRmKernels):
    - 10 warmup iterations
    - 100 timed iterations
    - Report: median, min, p95, mean, std (microseconds)
    - Output: JSON with structured results
"""

import argparse
import json
import sys

import torch
import numpy as np

import sm120_expert_kernels as EK

WARMUP_ITERS = 50
TIMED_ITERS = 500

# Model dimensions
HIDDEN_DIM = 7168
INTER_DIM = 2048
NUM_EXPERTS = 256
TOPK = 8         # V3.2
V4_TOPK = 6      # V4
TOKEN_COUNTS = [8, 64, 256, 1024]


def benchmark_kernel(name: str, fn, warmup=WARMUP_ITERS, iters=TIMED_ITERS,
                      verbose=False):
    """Benchmark a CUDA kernel using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    times = np.array(times_us)

    result = {
        "name": name,
        "median_us": float(np.median(times)),
        "min_us": float(np.min(times)),
        "p95_us": float(np.percentile(times, 95)),
        "mean_us": float(np.mean(times)),
        "std_us": float(np.std(times)),
    }

    if verbose:
        print(f"  {name}: median={result['median_us']:.1f} us, "
              f"min={result['min_us']:.1f}, p95={result['p95_us']:.1f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark SM120 Expert Kernels")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU available")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"SM capability: {torch.cuda.get_device_capability()}")
    print()

    results = {"gpu": gpu_name, "benchmarks": []}

    # ── SwiGLU (unclamped, V3.2) ────────────────────────────────────────────
    print("=== SwiGLU (unclamped, V3.2) ===")
    for N in TOKEN_COUNTS:
        expanded = N * TOPK
        inp = torch.randn(expanded, 2 * INTER_DIM,
                           dtype=torch.bfloat16, device="cuda")
        results["benchmarks"].append(
            benchmark_kernel(
                f"swiglu_unclamped_tokens{expanded}_d{INTER_DIM}",
                lambda: EK.fused_swiglu(inp),
                verbose=args.verbose))

    # ── SwiGLU (clamped, V4, limit=10.0) ─────────────────────────────────
    print("=== SwiGLU (clamped, V4, limit=10) ===")
    for N in TOKEN_COUNTS:
        expanded = N * V4_TOPK
        inp = torch.randn(expanded, 2 * INTER_DIM,
                           dtype=torch.bfloat16, device="cuda")
        results["benchmarks"].append(
            benchmark_kernel(
                f"swiglu_clamped10_tokens{expanded}_d{INTER_DIM}",
                lambda: EK.fused_swiglu(inp, swiglu_limit=10.0),
                verbose=args.verbose))

    # ── MoE Permute ───────────────────────────────────────────────────────────
    print("=== MoE Permute ===")
    for N in TOKEN_COUNTS:
        tokens = torch.randn(N, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
        topk_idx = torch.randint(0, NUM_EXPERTS, (N, TOPK),
                                  dtype=torch.int32, device="cuda")
        results["benchmarks"].append(
            benchmark_kernel(
                f"moe_permute_tokens{N}_e{NUM_EXPERTS}_k{TOPK}",
                lambda: EK.moe_permute(tokens, topk_idx, NUM_EXPERTS),
                verbose=args.verbose))

    # ── MoE Unpermute ─────────────────────────────────────────────────────────
    print("=== MoE Unpermute ===")
    for N in TOKEN_COUNTS:
        expanded = N * TOPK
        permuted = torch.randn(expanded, HIDDEN_DIM,
                                dtype=torch.bfloat16, device="cuda")
        weights = torch.rand(N, TOPK, dtype=torch.float32, device="cuda")
        s2d = torch.arange(expanded, dtype=torch.int32, device="cuda")
        results["benchmarks"].append(
            benchmark_kernel(
                f"moe_unpermute_tokens{N}_k{TOPK}",
                lambda: EK.moe_unpermute(permuted, weights, s2d, N, TOPK),
                verbose=args.verbose))

    # ── Top-K Gating (grouped, V3.2, sigmoid) ──────────────────────────────
    print("=== Top-K Gating (grouped, V3.2, sigmoid) ===")
    for N in TOKEN_COUNTS:
        logits = torch.randn(N, NUM_EXPERTS, dtype=torch.float32, device="cuda")
        bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
        results["benchmarks"].append(
            benchmark_kernel(
                f"topk_gating_grouped_sigmoid_tokens{N}_e{NUM_EXPERTS}",
                lambda: EK.topk_gating(logits, bias, topk=TOPK,
                                        n_group=8, topk_group=4,
                                        routed_scaling_factor=2.5,
                                        renormalize=True),
                verbose=args.verbose))

    # ── Top-K Gating (simple, sigmoid) ──────────────────────────────────────
    print("=== Top-K Gating (simple, sigmoid) ===")
    for N in TOKEN_COUNTS:
        logits = torch.randn(N, NUM_EXPERTS, dtype=torch.float32, device="cuda")
        bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
        results["benchmarks"].append(
            benchmark_kernel(
                f"topk_gating_simple_sigmoid_tokens{N}_e{NUM_EXPERTS}",
                lambda: EK.topk_gating(logits, bias, topk=TOPK,
                                        n_group=1, topk_group=1,
                                        routed_scaling_factor=1.0,
                                        renormalize=False),
                verbose=args.verbose))

    # ── Top-K Gating (V4: sqrtsoftplus, top-6, flat) ────────────────────────
    print("=== Top-K Gating (V4, sqrtsoftplus, top-6) ===")
    for N in TOKEN_COUNTS:
        logits = torch.randn(N, NUM_EXPERTS, dtype=torch.float32, device="cuda")
        bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
        results["benchmarks"].append(
            benchmark_kernel(
                f"topk_gating_sqrtsoftplus_tokens{N}_e{NUM_EXPERTS}",
                lambda: EK.topk_gating(logits, bias, topk=V4_TOPK,
                                        n_group=1, topk_group=1,
                                        routed_scaling_factor=2.5,
                                        renormalize=True,
                                        scoring_func="sqrtsoftplus"),
                verbose=args.verbose))

    # ── Adaptive Gating ───────────────────────────────────────────────────────
    print("=== Adaptive Gating ===")
    for N in TOKEN_COUNTS:
        w = torch.rand(N, TOPK, dtype=torch.float32, device="cuda")
        idx = torch.randint(0, NUM_EXPERTS, (N, TOPK),
                             dtype=torch.int32, device="cuda")
        results["benchmarks"].append(
            benchmark_kernel(
                f"adaptive_gating_tokens{N}_k{TOPK}",
                lambda: EK.adaptive_gating(w, idx, 0.9),
                verbose=args.verbose))

    # GEMM and FP8 quantization benchmarks are in sm120_gemm_kernels
    # (LayerStoRmGemmKernels). This module benchmarks only expert/MoE kernels.

    print()
    print(f"Total benchmarks: {len(results['benchmarks'])}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

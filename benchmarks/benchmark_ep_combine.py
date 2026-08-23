"""
Isolated EP-combine kernel microbenchmark (DET-REDUCE Phase 1b).

Times the unpermute + slot-reduce kernels of the three MoE EP-combine variants on
IDENTICAL input, so the combine cost is measured directly (not confounded by the
end-to-end routing trajectory):

  OFF            : finalize_moe_routing_bf16  (legacy reduced bf16, 1 kernel)
  canonical-fp32 : moe_unpermute_fp32 + moe_combine_reduce_slots_fp32_to_bf16
  canonical-bf16 : moe_unpermute_bf16_perslot + moe_combine_reduce_slots_bf16_to_bf16

Also reports the cross-GPU GATHER bytes per combine (analytic): the per-slot gather
is [B, topk, H] (fp32=4B, bf16=2B); the legacy reduced path gathers only [B, H] bf16.

Usage:  CUDA_VISIBLE_DEVICES=<one 5090> python benchmarks/benchmark_ep_combine.py
"""
import argparse
import json
import sys

import torch
import sm120_expert_kernels as EK

WARMUP = 50
ITERS = 500
HIDDEN = 7168
NUM_EXPERTS = 256
TOPK = 8
# Keeper decode is B=1; add a few larger B to expose the bandwidth trend.
TOKEN_COUNTS = [1, 8, 64, 256]


def time_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))  # ms->us
    n = len(us)
    return {
        "median_us": us[n // 2],
        "min_us": us[0],
        "p95_us": us[int(n * 0.95)],
        "mean_us": sum(us) / n,
    }


def bench_B(B, verbose=False):
    dev = "cuda"
    tokens = torch.randn(B, HIDDEN, dtype=torch.bfloat16, device=dev)
    topk_idx = torch.randint(0, NUM_EXPERTS, (B, TOPK), dtype=torch.int32, device=dev)
    permuted, offsets, s2d, perm_idx = EK.moe_permute(tokens, topk_idx, NUM_EXPERTS)
    weights = torch.rand(B, TOPK, dtype=torch.float32, device=dev)

    # OFF: legacy reduced bf16 (single fused unpermute+reduce kernel).
    off = time_fn(lambda: EK.moe_unpermute(permuted, weights, s2d, B, TOPK))

    # canonical-fp32: per-slot fp32 unpermute + fixed-order fp32 reduce.
    perslot_fp32 = EK.moe_unpermute_fp32(permuted, weights, s2d, B, TOPK)
    fp32_unp = time_fn(lambda: EK.moe_unpermute_fp32(permuted, weights, s2d, B, TOPK))
    fp32_red = time_fn(lambda: EK.moe_combine_reduce_slots_fp32_to_bf16(perslot_fp32))

    # canonical-bf16: per-slot bf16 unpermute + fixed-order fp32-accumulate reduce.
    perslot_bf16 = EK.moe_unpermute_bf16_perslot(permuted, weights, s2d, B, TOPK)
    bf16_unp = time_fn(lambda: EK.moe_unpermute_bf16_perslot(permuted, weights, s2d, B, TOPK))
    bf16_red = time_fn(lambda: EK.moe_combine_reduce_slots_bf16_to_bf16(perslot_bf16))

    gather_fp32 = B * TOPK * HIDDEN * 4
    gather_bf16 = B * TOPK * HIDDEN * 2
    gather_off = B * HIDDEN * 2

    res = {
        "B": B,
        "off": {"kernel_us": off["median_us"], "gather_bytes": gather_off},
        "canonical_fp32": {
            "unpermute_us": fp32_unp["median_us"],
            "reduce_us": fp32_red["median_us"],
            "combine_us": fp32_unp["median_us"] + fp32_red["median_us"],
            "gather_bytes": gather_fp32,
        },
        "canonical_bf16": {
            "unpermute_us": bf16_unp["median_us"],
            "reduce_us": bf16_red["median_us"],
            "combine_us": bf16_unp["median_us"] + bf16_red["median_us"],
            "gather_bytes": gather_bf16,
        },
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        sys.exit(1)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"H={HIDDEN} E={NUM_EXPERTS} topk={TOPK}  warmup={WARMUP} iters={ITERS}\n")
    all_res = []
    hdr = (f"{'B':>5} | {'OFF kern':>9} | {'fp32 unp':>9} {'fp32 red':>9} {'fp32 comb':>10}"
           f" | {'bf16 unp':>9} {'bf16 red':>9} {'bf16 comb':>10}"
           f" | {'gather fp32':>12} {'gather bf16':>12} {'bf16/fp32':>9}")
    print(hdr)
    print("-" * len(hdr))
    for B in TOKEN_COUNTS:
        r = bench_B(B)
        all_res.append(r)
        f, b = r["canonical_fp32"], r["canonical_bf16"]
        ratio = b["combine_us"] / f["combine_us"] if f["combine_us"] else 0
        print(f"{B:>5} | {r['off']['kernel_us']:>9.2f} | "
              f"{f['unpermute_us']:>9.2f} {f['reduce_us']:>9.2f} {f['combine_us']:>10.2f} | "
              f"{b['unpermute_us']:>9.2f} {b['reduce_us']:>9.2f} {b['combine_us']:>10.2f} | "
              f"{f['gather_bytes']:>12} {b['gather_bytes']:>12} {ratio:>8.2f}x")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(all_res, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

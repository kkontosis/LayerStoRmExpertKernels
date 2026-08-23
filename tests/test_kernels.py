"""
GPU kernel tests for SM120 Expert kernels.
Requires: SM120 GPU + sm120_expert_kernels built.

Run: CUDA_VISIBLE_DEVICES=0 pytest tests/test_kernels.py -v
"""

import sys
import os
import unittest

import torch
import torch.nn.functional as F

# Ensure tests/ parent is on path so we can import reference implementations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_reference import (
    ref_swiglu, ref_adaptive_gating, ref_topk_gating,
)

import sm120_expert_kernels as EK


def has_cuda():
    return torch.cuda.is_available()


def has_sm120():
    if not has_cuda():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] > 12 or (cap[0] == 12 and cap[1] >= 0)


requires_gpu = unittest.skipUnless(has_cuda(), "Requires CUDA GPU")


class TestExpertKernels(unittest.TestCase):
    """GPU kernel tests — compare CUDA output against PyTorch reference."""

    @requires_gpu
    def test_kernel_fused_swiglu(self):
        """SwiGLU kernel: SiLU(gate) * up, BF16."""
        N, D = 256, 2048
        gate = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")
        up = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")

        # Kernel expects interleaved [gate | up] input
        inp = torch.cat([gate, up], dim=1)  # [N, 2*D]
        output = EK.fused_swiglu(inp)

        ref = ref_swiglu(gate.cpu().float(), up.cpu().float())
        cosine = F.cosine_similarity(
            output.cpu().float().flatten(), ref.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.9999,
                           f"SwiGLU cosine={cosine:.6f}")

    @requires_gpu
    def test_kernel_moe_permute_unpermute(self):
        """Permute + unpermute round-trip must recover original data."""
        B, D = 128, 512
        num_experts = 32
        topk = 4

        tokens = torch.randn(B, D, dtype=torch.bfloat16, device="cuda")
        topk_indices = torch.randint(0, num_experts, (B, topk),
                                      dtype=torch.int32, device="cuda")

        # Permute
        permuted, offsets, s2d_map, perm_idx = EK.moe_permute(
            tokens, topk_indices, num_experts)

        self.assertEqual(permuted.shape, (B * topk, D))
        self.assertEqual(offsets.shape[0], num_experts + 1)

        # Unpermute with uniform weights (1/topk each)
        weights = torch.full((B, topk), 1.0 / topk,
                              dtype=torch.float32, device="cuda")
        recovered = EK.moe_unpermute(permuted, weights, s2d_map, B, topk)

        self.assertEqual(recovered.shape, (B, D))

        # Each original token appears topk times in permuted output.
        # Unpermute with 1/topk weights should give back the original.
        max_diff = (recovered.float() - tokens.float()).abs().max().item()
        # BF16 arithmetic introduces some rounding
        self.assertLess(max_diff, 0.02,
                        f"Permute roundtrip max_diff={max_diff}")

    @requires_gpu
    def test_kernel_moe_unpermute_fp32_perslot_and_reduce(self):
        """DET-REDUCE Phase 1b (canonical): per-slot fp32 unpermute must write each
        of the K expert contributions to its own slot exactly; the fixed-order
        K-slot reduce must equal the bf16 of that fp32 slot-sum; and — the
        placement-invariance property — summing the slots in ANY split/order yields
        the SAME bf16 result (the canonical reduce is grouping-independent only when
        the final sum is fixed-order, which the kernel guarantees)."""
        B, D = 96, 7168           # D divisible by 8 → exercises the float4 store path
        num_experts = 64
        topk = 8

        tokens = torch.randn(B, D, dtype=torch.bfloat16, device="cuda")
        topk_indices = torch.randint(0, num_experts, (B, topk),
                                      dtype=torch.int32, device="cuda")
        permuted, offsets, s2d_map, perm_idx = EK.moe_permute(
            tokens, topk_indices, num_experts)
        weights = torch.rand(B, topk, dtype=torch.float32, device="cuda")

        perslot = EK.moe_unpermute_fp32(permuted, weights, s2d_map, B, topk)
        self.assertEqual(perslot.dtype, torch.float32)
        self.assertEqual(perslot.shape, (B, topk, D))

        # Per-slot reference: slot k = weight_k * expert_row_k (bf16→fp32). The
        # kernel multiplies bf16→fp32 then by the fp32 weight, identical to this.
        s2d = s2d_map.view(B, topk)
        ref_slots = torch.zeros(B, topk, D, dtype=torch.float32, device="cuda")
        for k in range(topk):
            ref_slots[:, k, :] = weights[:, k:k+1] * permuted[s2d[:, k].long()].float()
        self.assertTrue(torch.equal(perslot, ref_slots),
                        "per-slot fp32 contributions must match the reference exactly")

        # Fixed-order K-slot reduce → bf16 must equal bf16(sum_k slots in order).
        out = EK.moe_combine_reduce_slots_fp32_to_bf16(perslot.contiguous())
        self.assertEqual(out.dtype, torch.bfloat16)
        ref_sum = torch.zeros(B, D, dtype=torch.float32, device="cuda")
        for k in range(topk):
            ref_sum += perslot[:, k, :]
        self.assertTrue(torch.equal(out, ref_sum.to(torch.bfloat16)),
                        "fixed-order reduce must equal bf16 of the in-order fp32 sum")

        # Placement-invariance: the cross-GPU gather is modeled by splitting the K
        # slots into two disjoint partitions (any split), zero-filling the absent
        # slots in each, summing the two per-slot tensors (the allreduce), then
        # reducing. The bf16 result must be IDENTICAL regardless of the split,
        # because the final fixed-order sum is over the same per-slot values.
        for split in (1, 3, 5):
            a = perslot.clone(); a[:, split:, :] = 0.0
            b = perslot.clone(); b[:, :split, :] = 0.0
            gathered = a + b                                   # allreduce-sum
            out_split = EK.moe_combine_reduce_slots_fp32_to_bf16(gathered.contiguous())
            self.assertTrue(torch.equal(out_split, out),
                            f"canonical combine must be placement-invariant (split={split})")

    @requires_gpu
    def test_kernel_moe_unpermute_bf16_perslot_and_reduce(self):
        """DET-REDUCE Phase 1b (canonical, BF16 payload): per-slot bf16 unpermute must
        write each of the K expert contributions to its own slot as bf16(weight_k *
        expert_row_k); the fixed-order K-slot reduce must equal the bf16 of the
        fp32 in-order slot-sum; and — placement-invariance — summing the bf16 slots
        in ANY split/order (the bf16 SUM-allreduce gather is exact since 0+x=x)
        yields the SAME bf16 result. Invariance comes from the fixed slot ORDER,
        not the payload dtype."""
        B, D = 96, 7168           # D divisible by 8 → exercises the int4 store path
        num_experts = 64
        topk = 8

        tokens = torch.randn(B, D, dtype=torch.bfloat16, device="cuda")
        topk_indices = torch.randint(0, num_experts, (B, topk),
                                      dtype=torch.int32, device="cuda")
        permuted, offsets, s2d_map, perm_idx = EK.moe_permute(
            tokens, topk_indices, num_experts)
        weights = torch.rand(B, topk, dtype=torch.float32, device="cuda")

        perslot = EK.moe_unpermute_bf16_perslot(permuted, weights, s2d_map, B, topk)
        self.assertEqual(perslot.dtype, torch.bfloat16)
        self.assertEqual(perslot.shape, (B, topk, D))

        # Per-slot reference: slot k = bf16(weight_k * (bf16→fp32 expert_row_k)).
        # The kernel multiplies bf16→fp32 by the fp32 weight then rounds to bf16 once.
        s2d = s2d_map.view(B, topk)
        ref_slots = torch.zeros(B, topk, D, dtype=torch.bfloat16, device="cuda")
        for k in range(topk):
            prod = weights[:, k:k+1] * permuted[s2d[:, k].long()].float()
            ref_slots[:, k, :] = prod.to(torch.bfloat16)
        self.assertTrue(torch.equal(perslot, ref_slots),
                        "per-slot bf16 contributions must match the reference exactly")

        # Fixed-order K-slot reduce → bf16 must equal bf16(sum_k slots in order, fp32).
        out = EK.moe_combine_reduce_slots_bf16_to_bf16(perslot.contiguous())
        self.assertEqual(out.dtype, torch.bfloat16)
        ref_sum = torch.zeros(B, D, dtype=torch.float32, device="cuda")
        for k in range(topk):
            ref_sum += perslot[:, k, :].float()
        self.assertTrue(torch.equal(out, ref_sum.to(torch.bfloat16)),
                        "fixed-order bf16 reduce must equal bf16 of the in-order fp32 sum")

        # Placement-invariance: model the cross-GPU bf16 SUM-allreduce gather by
        # splitting the K slots into two disjoint partitions, zero-filling the absent
        # slots in each, summing the two bf16 per-slot tensors (the allreduce; 0+x=x
        # is exact in bf16), then reducing. The result must be IDENTICAL regardless
        # of the split — invariance is from the fixed slot ORDER, not the payload.
        for split in (1, 3, 5):
            a = perslot.clone(); a[:, split:, :] = 0.0
            b = perslot.clone(); b[:, :split, :] = 0.0
            gathered = a + b                                   # bf16 allreduce-sum
            self.assertTrue(torch.equal(gathered, perslot),
                            f"bf16 gather must be exact (0+x=x) (split={split})")
            out_split = EK.moe_combine_reduce_slots_bf16_to_bf16(gathered.contiguous())
            self.assertTrue(torch.equal(out_split, out),
                            f"bf16-payload combine must be placement-invariant (split={split})")

    @requires_gpu
    def test_kernel_adaptive_gating(self):
        """Adaptive gating: threshold pruning matches CPU reference."""
        num_tokens, topk = 64, 8
        threshold = 0.9

        in_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device="cuda")
        in_indices = torch.randint(0, 256, (num_tokens, topk),
                                    dtype=torch.int32, device="cuda")

        out_w, out_i, counts = EK.adaptive_gating(in_weights, in_indices, threshold)

        # CPU reference
        ref_w, ref_i, ref_c = ref_adaptive_gating(
            in_weights.cpu(), in_indices.cpu(), threshold)

        # Counts should match exactly
        self.assertTrue(torch.equal(counts.cpu(), ref_c),
                        "Adaptive gating: expert counts mismatch")

        # Weights and indices should match for kept experts
        for t in range(num_tokens):
            c = ref_c[t].item()
            self.assertTrue(
                torch.allclose(out_w[t, :c].cpu(), ref_w[t, :c], atol=1e-6),
                f"Token {t}: weight mismatch")
            self.assertTrue(
                torch.equal(out_i[t, :c].cpu(), ref_i[t, :c]),
                f"Token {t}: index mismatch")

    @requires_gpu
    def test_kernel_topk_gating_simple(self):
        """Top-K gating (simple, n_group=1) matches CPU reference."""
        num_tokens, num_experts, topk = 64, 256, 8

        logits = torch.randn(num_tokens, num_experts,
                              dtype=torch.float32, device="cuda")
        bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1

        weights, indices = EK.topk_gating(
            logits, bias, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True)

        ref_w, ref_i = ref_topk_gating(
            logits.cpu(), bias.cpu(), topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True)

        # Expert selections should match (same top-K on same scores)
        for t in range(num_tokens):
            kernel_set = set(indices[t].cpu().tolist())
            ref_set = set(ref_i[t].tolist())
            self.assertEqual(kernel_set, ref_set,
                             f"Token {t}: index set mismatch")

        # Weight cosine should be very high
        cosine = F.cosine_similarity(
            weights.cpu().float().flatten(), ref_w.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.9999,
                           f"TopK gating weight cosine={cosine:.6f}")

    @requires_gpu
    def test_kernel_topk_gating_grouped(self):
        """Top-K gating (grouped, V3.2 style) selects from correct groups."""
        num_tokens = 32
        num_experts = 256
        topk = 8
        n_group = 8
        topk_group = 4
        experts_per_group = num_experts // n_group

        logits = torch.randn(num_tokens, num_experts,
                              dtype=torch.float32, device="cuda")
        bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1

        weights, indices = EK.topk_gating(
            logits, bias, topk=topk, n_group=n_group, topk_group=topk_group,
            routed_scaling_factor=2.5, renormalize=True)

        # Verify: selected experts come from at most topk_group groups
        for t in range(num_tokens):
            groups_used = set()
            for k in range(topk):
                groups_used.add(indices[t, k].item() // experts_per_group)
            self.assertLessEqual(len(groups_used), topk_group,
                                 f"Token {t}: used {len(groups_used)} groups")

        # Verify renormalized weights sum to scaling_factor
        for t in range(num_tokens):
            w_sum = weights[t].sum().item()
            self.assertAlmostEqual(w_sum, 2.5, places=2,
                                   msg=f"Token {t}: weight sum={w_sum}")

    @requires_gpu
    def test_kernel_topk_gating_no_bias(self):
        """Top-K gating works with bias=None."""
        num_tokens, num_experts, topk = 16, 64, 4

        logits = torch.randn(num_tokens, num_experts,
                              dtype=torch.float32, device="cuda")

        weights, indices = EK.topk_gating(
            logits, None, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=1.0, renormalize=False)

        self.assertEqual(weights.shape, (num_tokens, topk))
        self.assertEqual(indices.shape, (num_tokens, topk))

        # Without bias, selection is pure sigmoid top-K
        scores = torch.sigmoid(logits.cpu().float())
        _, manual_idx = scores.topk(topk, dim=1)
        for t in range(num_tokens):
            self.assertEqual(
                set(indices[t].cpu().tolist()),
                set(manual_idx[t].tolist()),
                f"Token {t}: mismatch with no-bias topk")

    @requires_gpu
    def test_edge_zero_tokens(self):
        """Zero-token inputs should not crash."""
        inp = torch.empty(0, 128, dtype=torch.bfloat16, device="cuda")
        out = EK.fused_swiglu(inp)
        self.assertEqual(out.shape, (0, 64))

    @requires_gpu
    def test_edge_single_token(self):
        """Single-token inputs should work correctly."""
        D = 512
        inp = torch.randn(1, 2 * D, dtype=torch.bfloat16, device="cuda")
        out = EK.fused_swiglu(inp)
        self.assertEqual(out.shape, (1, D))

        ref = ref_swiglu(inp[:, :D].cpu().float(), inp[:, D:].cpu().float())
        cosine = F.cosine_similarity(
            out.cpu().float().flatten(), ref.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.999)


if __name__ == "__main__":
    unittest.main()

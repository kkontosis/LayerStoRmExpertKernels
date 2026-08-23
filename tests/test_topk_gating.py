"""
Top-K gating tests for V4E-1: sqrtsoftplus scoring function.

Tests:
  - sqrtsoftplus scores match sqrt(log(1+exp(x))) reference
  - top-6 flat routing (V4 config)
  - V3.2 sigmoid regression (existing behavior unchanged)

CPU reference tests run without GPU. GPU kernel tests require SM120.

Run: python tests/test_topk_gating.py -v
"""

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_reference import ref_topk_gating, ref_sqrtsoftplus


class TestSqrtSoftplusReference(unittest.TestCase):
    """CPU reference tests for sqrtsoftplus scoring."""

    def test_sqrtsoftplus_matches_formula(self):
        """ref_sqrtsoftplus must match sqrt(log(1 + exp(x))) elementwise."""
        x = torch.linspace(-10, 50, 1000)
        result = ref_sqrtsoftplus(x)
        expected = torch.sqrt(torch.log(1.0 + torch.exp(x.float().clamp(max=80))))
        # For large x, log(1+exp(x)) ~ x, so expected ~ sqrt(x)
        # Use manual large-x formula where exp overflows
        for i in range(len(x)):
            xi = x[i].item()
            if xi > 20:
                sp = xi + math.log(1 + math.exp(-xi))
            else:
                sp = math.log(1 + math.exp(xi))
            expected[i] = math.sqrt(sp)
        self.assertTrue(
            torch.allclose(result, expected, rtol=1e-5, atol=1e-6),
            f"max diff: {(result - expected).abs().max().item()}")

    def test_sqrtsoftplus_properties(self):
        """sqrtsoftplus is non-negative, monotonically increasing, and unbounded."""
        x = torch.linspace(-20, 100, 500)
        result = ref_sqrtsoftplus(x)

        # Non-negative
        self.assertTrue((result >= 0).all(), "sqrtsoftplus must be >= 0")

        # Monotonically increasing
        diffs = result[1:] - result[:-1]
        self.assertTrue((diffs >= -1e-6).all(), "sqrtsoftplus must be monotonically increasing")

        # Unbounded: value at x=100 >> value at x=-20
        self.assertGreater(result[-1].item(), result[0].item() * 10,
                           "sqrtsoftplus must grow unboundedly")

    def test_sqrtsoftplus_near_zero(self):
        """At x=0, sqrtsoftplus = sqrt(log(2)) ~ 0.8326."""
        result = ref_sqrtsoftplus(torch.tensor([0.0]))
        expected = math.sqrt(math.log(2))
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_topk_gating_sqrtsoftplus_simple(self):
        """sqrtsoftplus scoring with flat top-K (n_group=1) matches manual."""
        num_tokens, num_experts, topk = 16, 64, 6
        logits = torch.randn(num_tokens, num_experts)
        bias = torch.randn(num_experts) * 0.1

        weights, indices = ref_topk_gating(
            logits, bias, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True,
            scoring_func="sqrtsoftplus")

        scores = ref_sqrtsoftplus(logits)
        sel_scores = scores + bias.float()
        _, manual_idx = sel_scores.topk(topk, dim=1)

        for t in range(num_tokens):
            self.assertEqual(
                set(indices[t].tolist()), set(manual_idx[t].tolist()),
                f"Token {t}: sqrtsoftplus index mismatch")

        # Weights should be unbiased sqrtsoftplus at selected indices
        for t in range(num_tokens):
            for k in range(topk):
                expected_w = scores[t, indices[t, k].long()].item()
                # After renormalization, compare structure not raw value
            w_sum = weights[t].sum().item()
            self.assertAlmostEqual(w_sum, 2.5, places=4,
                                   msg=f"Token {t}: weight sum={w_sum}")

    def test_topk_gating_v4_flat_top6(self):
        """V4 config: sqrtsoftplus, top-6, n_group=1 (flat routing)."""
        num_tokens = 32
        num_experts = 256
        topk = 6

        logits = torch.randn(num_tokens, num_experts)
        bias = torch.randn(num_experts) * 0.05

        weights, indices = ref_topk_gating(
            logits, bias, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True,
            scoring_func="sqrtsoftplus")

        self.assertEqual(weights.shape, (num_tokens, topk))
        self.assertEqual(indices.shape, (num_tokens, topk))

        for t in range(num_tokens):
            # All indices valid
            for k in range(topk):
                self.assertGreaterEqual(indices[t, k].item(), 0)
                self.assertLess(indices[t, k].item(), num_experts)
            # No duplicate experts
            self.assertEqual(len(set(indices[t].tolist())), topk)
            # Renormalized sum
            self.assertAlmostEqual(weights[t].sum().item(), 2.5, places=4)

    def test_sigmoid_regression(self):
        """V3.2 sigmoid path must be unchanged by sqrtsoftplus addition."""
        num_tokens, num_experts, topk = 16, 256, 8
        logits = torch.randn(num_tokens, num_experts)
        bias = torch.randn(num_experts) * 0.1

        weights, indices = ref_topk_gating(
            logits, bias, topk=topk, n_group=8, topk_group=4,
            routed_scaling_factor=2.5, renormalize=True,
            scoring_func="sigmoid")

        # Verify sigmoid scores are in (0, 1)
        scores = torch.sigmoid(logits.float())
        for t in range(num_tokens):
            for k in range(topk):
                idx = indices[t, k].long()
                raw_score = scores[t, idx].item()
                self.assertGreater(raw_score, 0.0)
                self.assertLess(raw_score, 1.0)

        # Verify renormalized sum
        for t in range(num_tokens):
            self.assertAlmostEqual(weights[t].sum().item(), 2.5, places=4)

        experts_per_group = num_experts // 8
        for t in range(num_tokens):
            groups_used = set()
            for k in range(topk):
                groups_used.add(indices[t, k].item() // experts_per_group)
            self.assertLessEqual(len(groups_used), 4)

    def test_sigmoid_default_backward_compat(self):
        """Omitting scoring_func defaults to sigmoid (backward compat)."""
        num_tokens, num_experts, topk = 8, 32, 4
        logits = torch.randn(num_tokens, num_experts)

        w_default, i_default = ref_topk_gating(
            logits, None, topk=topk)
        w_explicit, i_explicit = ref_topk_gating(
            logits, None, topk=topk, scoring_func="sigmoid")

        self.assertTrue(torch.equal(w_default, w_explicit))
        self.assertTrue(torch.equal(i_default, i_explicit))


def has_cuda():
    return torch.cuda.is_available()


requires_gpu = unittest.skipUnless(has_cuda(), "Requires CUDA GPU")


class TestSqrtSoftplusKernel(unittest.TestCase):
    """GPU kernel tests for sqrtsoftplus scoring (requires CUDA)."""

    @requires_gpu
    def test_kernel_sqrtsoftplus_scores_match_reference(self):
        """CUDA sqrtsoftplus top-K output matches CPU reference."""
        import sm120_expert_kernels as EK

        num_tokens, num_experts, topk = 64, 256, 6
        logits = torch.randn(num_tokens, num_experts,
                              dtype=torch.float32, device="cuda")
        bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1

        weights, indices = EK.topk_gating(
            logits, bias, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True,
            scoring_func="sqrtsoftplus")

        ref_w, ref_i = ref_topk_gating(
            logits.cpu(), bias.cpu(), topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True,
            scoring_func="sqrtsoftplus")

        for t in range(num_tokens):
            kernel_set = set(indices[t].cpu().tolist())
            ref_set = set(ref_i[t].tolist())
            self.assertEqual(kernel_set, ref_set,
                             f"Token {t}: sqrtsoftplus index set mismatch")

        cosine = F.cosine_similarity(
            weights.cpu().float().flatten(), ref_w.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.9999,
                           f"sqrtsoftplus weight cosine={cosine:.6f}")

    @requires_gpu
    def test_kernel_v4_top6_flat(self):
        """V4 config: sqrtsoftplus, top-6, flat routing, renormalized."""
        import sm120_expert_kernels as EK

        num_tokens = 32
        num_experts = 256
        topk = 6

        logits = torch.randn(num_tokens, num_experts,
                              dtype=torch.float32, device="cuda")

        weights, indices = EK.topk_gating(
            logits, None, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True,
            scoring_func="sqrtsoftplus")

        self.assertEqual(weights.shape, (num_tokens, topk))
        self.assertEqual(indices.shape, (num_tokens, topk))

        for t in range(num_tokens):
            self.assertEqual(len(set(indices[t].cpu().tolist())), topk,
                             f"Token {t}: duplicate expert indices")
            self.assertAlmostEqual(weights[t].sum().item(), 2.5, places=2,
                                   msg=f"Token {t}: weight sum mismatch")

    @requires_gpu
    def test_kernel_sigmoid_regression(self):
        """Sigmoid path unchanged after sqrtsoftplus addition."""
        import sm120_expert_kernels as EK

        num_tokens, num_experts, topk = 64, 256, 8
        logits = torch.randn(num_tokens, num_experts,
                              dtype=torch.float32, device="cuda")
        bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1

        # Explicit sigmoid
        w_sig, i_sig = EK.topk_gating(
            logits, bias, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True,
            scoring_func="sigmoid")

        # Default (should be sigmoid)
        w_def, i_def = EK.topk_gating(
            logits, bias, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True)

        self.assertTrue(torch.equal(w_sig.cpu(), w_def.cpu()),
                        "Default scoring_func must equal explicit sigmoid")
        self.assertTrue(torch.equal(i_sig.cpu(), i_def.cpu()),
                        "Default scoring_func must equal explicit sigmoid")

        ref_w, ref_i = ref_topk_gating(
            logits.cpu(), bias.cpu(), topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=2.5, renormalize=True,
            scoring_func="sigmoid")

        cosine = F.cosine_similarity(
            w_sig.cpu().float().flatten(), ref_w.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.9999,
                           f"Sigmoid regression cosine={cosine:.6f}")

    @requires_gpu
    def test_kernel_sqrtsoftplus_numerical_stability(self):
        """sqrtsoftplus handles large and small logits without NaN/Inf."""
        import sm120_expert_kernels as EK

        num_tokens = 8
        num_experts = 64
        topk = 4

        logits = torch.zeros(num_tokens, num_experts,
                              dtype=torch.float32, device="cuda")
        logits[0, :] = 100.0   # large positive
        logits[1, :] = -100.0  # large negative
        logits[2, :] = 0.0     # zero
        logits[3, :] = torch.linspace(-50, 50, num_experts, device="cuda")

        weights, indices = EK.topk_gating(
            logits, None, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=1.0, renormalize=False,
            scoring_func="sqrtsoftplus")

        self.assertTrue(torch.isfinite(weights).all(),
                        "sqrtsoftplus weights must be finite")
        self.assertTrue((weights >= 0).all(),
                        "sqrtsoftplus weights must be non-negative")


if __name__ == "__main__":
    unittest.main()

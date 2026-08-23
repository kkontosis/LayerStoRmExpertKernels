"""
Fused SwiGLU tests for V4E-2: swiglu_limit clamping.

Tests:
  - Clamped output bounded by +/- limit
  - V3.2 unclamped regression (swiglu_limit=0.0)

CPU reference tests run without GPU. GPU kernel tests require SM120.

Run: python tests/test_fused_swiglu.py -v
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_reference import ref_swiglu


class TestSwigluClampReference(unittest.TestCase):
    """CPU reference tests for SwiGLU clamping."""

    def test_unclamped_matches_silu(self):
        """swiglu_limit=0 must match plain SiLU(gate) * up."""
        N, D = 64, 512
        gate = torch.randn(N, D)
        up = torch.randn(N, D)

        result = ref_swiglu(gate, up, swiglu_limit=0.0)
        expected = F.silu(gate.float()) * up.float()
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-7))

    def test_clamped_gate_upper_bounded(self):
        """With limit=10, gate is capped at +10 before SiLU (upper bound only)."""
        N, D = 64, 512
        gate = torch.randn(N, D) * 20  # many values exceed 10
        up = torch.ones(N, D)
        limit = 10.0

        result = ref_swiglu(gate, up, swiglu_limit=limit)

        # SiLU(min(gate, 10)) is bounded above: SiLU(10) ~ 9.9995
        silu_max = F.silu(torch.tensor(limit)).item()
        self.assertTrue((result <= silu_max + 1e-5).all(),
                        f"Clamped output exceeds SiLU({limit})={silu_max:.4f}")

    def test_gate_lower_not_clamped(self):
        """llama.cpp DEEPSEEK4: gate has NO lower clamp — SiLU(-20) != SiLU(-10)."""
        limit = 10.0
        gate = torch.tensor([-20.0, -10.0])
        up = torch.ones(2)

        result = ref_swiglu(gate, up, swiglu_limit=limit)
        expected = F.silu(gate)  # untouched below the limit
        self.assertTrue(torch.allclose(result, expected, rtol=1e-6, atol=1e-9),
                        "gate must NOT be clamped at the lower end")
        # And they genuinely differ from a two-sided-clamp reference:
        two_sided = F.silu(gate.clamp(-limit, limit))
        self.assertFalse(torch.allclose(result, two_sided, rtol=0, atol=1e-9))

    def test_v4_limit_10(self):
        """V4 config: swiglu_limit=10.0 clamps large activations."""
        N, D = 256, 2048
        gate = torch.randn(N, D) * 15
        up = torch.randn(N, D)

        clamped = ref_swiglu(gate, up, swiglu_limit=10.0)
        unclamped = ref_swiglu(gate, up, swiglu_limit=0.0)

        # Clamped and unclamped should differ where |gate| > 10
        large_gate = gate.float().abs() > 10.0
        if large_gate.any():
            diff = (clamped - unclamped).abs()
            self.assertGreater(diff[large_gate].max().item(), 0.01,
                               "Clamping should change output where |gate| > limit")

        # Where |gate| <= 10, results should be very close
        small_gate = gate.float().abs() <= 9.0
        if small_gate.any():
            self.assertTrue(
                torch.allclose(clamped[small_gate], unclamped[small_gate],
                               rtol=1e-5, atol=1e-6),
                "Clamping should not affect outputs where |gate| < limit")

    def test_up_clamped_two_sided(self):
        """llama.cpp DEEPSEEK4: up IS clamped to [-limit, limit]."""
        gate = torch.tensor([1.0, 1.0, 1.0])
        up = torch.tensor([100.0, -100.0, 5.0])
        limit = 10.0

        result = ref_swiglu(gate, up, swiglu_limit=limit)
        expected = F.silu(gate) * up.clamp(-limit, limit)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-6, atol=1e-9),
                        "up must be clamped two-sided at +/- limit")
        # 100 and 5 must NOT scale 20x once up saturates at the limit:
        ratio = result[0].item() / result[2].item()
        self.assertAlmostEqual(ratio, limit / 5.0, places=3)

    def test_backward_compat_default(self):
        """Default swiglu_limit=0 produces same result as old ref_swiglu."""
        N, D = 32, 256
        gate = torch.randn(N, D)
        up = torch.randn(N, D)

        with_default = ref_swiglu(gate, up)
        with_zero = ref_swiglu(gate, up, swiglu_limit=0.0)
        plain_silu = F.silu(gate.float()) * up.float()

        self.assertTrue(torch.equal(with_default, with_zero))
        self.assertTrue(torch.allclose(with_default, plain_silu, rtol=1e-5, atol=1e-7))


def has_cuda():
    return torch.cuda.is_available()


requires_gpu = unittest.skipUnless(has_cuda(), "Requires CUDA GPU")


class TestSwigluClampKernel(unittest.TestCase):
    """GPU kernel tests for SwiGLU clamping (requires CUDA)."""

    @requires_gpu
    def test_kernel_clamped_matches_reference(self):
        """CUDA clamped SwiGLU matches CPU reference."""
        import sm120_expert_kernels as EK

        N, D = 256, 2048
        # Scale BOTH halves past the limit so the gate max-clamp AND the
        # two-sided up clamp are exercised (llama.cpp DEEPSEEK4 semantics).
        gate = torch.randn(N, D, dtype=torch.bfloat16, device="cuda") * 15
        up = torch.randn(N, D, dtype=torch.bfloat16, device="cuda") * 15
        inp = torch.cat([gate, up], dim=1)

        output = EK.fused_swiglu(inp, swiglu_limit=10.0)

        ref = ref_swiglu(gate.cpu().float(), up.cpu().float(), swiglu_limit=10.0)
        cosine = F.cosine_similarity(
            output.cpu().float().flatten(), ref.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.999,
                           f"Clamped SwiGLU cosine={cosine:.6f}")

    @requires_gpu
    def test_kernel_clamped_output_bounded(self):
        """Kernel output with limit=10 has SiLU component bounded."""
        import sm120_expert_kernels as EK

        N, D = 128, 1024
        gate = torch.randn(N, D, dtype=torch.bfloat16, device="cuda") * 20
        up = torch.ones(N, D, dtype=torch.bfloat16, device="cuda")
        inp = torch.cat([gate, up], dim=1)

        output = EK.fused_swiglu(inp, swiglu_limit=10.0)

        silu_max = F.silu(torch.tensor(10.0)).item()
        max_out = output.cpu().float().max().item()
        self.assertLessEqual(max_out, silu_max + 0.05,
                             f"Clamped output max={max_out} exceeds SiLU(10)={silu_max:.4f}")

    @requires_gpu
    def test_kernel_unclamped_regression(self):
        """swiglu_limit=0 (V3.2 default) matches unclamped kernel output."""
        import sm120_expert_kernels as EK

        N, D = 256, 2048
        gate = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")
        up = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")
        inp = torch.cat([gate, up], dim=1)

        out_default = EK.fused_swiglu(inp)
        out_zero = EK.fused_swiglu(inp, swiglu_limit=0.0)

        self.assertTrue(torch.equal(out_default, out_zero),
                        "Default swiglu_limit must equal explicit 0.0")

        ref = ref_swiglu(gate.cpu().float(), up.cpu().float())
        cosine = F.cosine_similarity(
            out_default.cpu().float().flatten(), ref.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.9999,
                           f"Unclamped SwiGLU regression cosine={cosine:.6f}")

    @requires_gpu
    def test_kernel_v4_limit_changes_output(self):
        """Clamped and unclamped produce different results on large inputs."""
        import sm120_expert_kernels as EK

        N, D = 64, 512
        gate = torch.randn(N, D, dtype=torch.bfloat16, device="cuda") * 15
        up = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")
        inp = torch.cat([gate, up], dim=1)

        out_clamped = EK.fused_swiglu(inp, swiglu_limit=10.0)
        out_unclamped = EK.fused_swiglu(inp, swiglu_limit=0.0)

        diff = (out_clamped.float() - out_unclamped.float()).abs().max().item()
        self.assertGreater(diff, 0.01,
                           "Clamping should change output for large activations")


if __name__ == "__main__":
    unittest.main()

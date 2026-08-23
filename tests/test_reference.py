"""
Pure-PyTorch reference tests for SM120 Expert kernels.
CPU only — no GPU required. Establishes error budgets for GPU kernel validation.

Run: python tests/test_reference.py -v
"""

import math
import unittest

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def ref_nvfp4_dequant(weights_packed: torch.Tensor, scales: torch.Tensor,
                       M: int, N: int, group_size: int = 16) -> torch.Tensor:
    """Reference NVFP4 dequantization (CPU, float32).

    Args:
        weights_packed: [M, N//2] uint8 — two FP4 values per byte
        scales: [M, N//group_size] float32 — UE8M0 block scales
        M, N: output shape
        group_size: elements per scale group (16)

    Returns:
        [M, N] float32 dequantized weights
    """
    # FP4 E2M1 lookup table (0..15 → float)
    fp4_lut = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=torch.float32)

    output = torch.zeros(M, N, dtype=torch.float32)
    for row in range(M):
        for col_pair in range(N // 2):
            byte_val = weights_packed[row, col_pair].item()
            lo = byte_val & 0x0F
            hi = (byte_val >> 4) & 0x0F

            col_lo = 2 * col_pair
            col_hi = 2 * col_pair + 1

            scale_lo = scales[row, col_lo // group_size].item()
            scale_hi = scales[row, col_hi // group_size].item()

            output[row, col_lo] = fp4_lut[lo] * scale_lo
            output[row, col_hi] = fp4_lut[hi] * scale_hi

    return output


def ref_fp8_dequant(data_fp8: torch.Tensor, scales: torch.Tensor,
                     M: int, N: int, block_size: int = 128) -> torch.Tensor:
    """Reference FP8 E4M3 dequantization with per-block scales (CPU, float32).

    Args:
        data_fp8: [M, N] float tensor (values in FP8 E4M3 range, already cast to float)
        scales: [M, ceil(N/block_size)] float32 per-block scales
        M, N: tensor shape
        block_size: elements per scale block (128)

    Returns:
        [M, N] float32 dequantized tensor
    """
    output = torch.zeros(M, N, dtype=torch.float32)
    for row in range(M):
        for col in range(N):
            block_idx = col // block_size
            output[row, col] = data_fp8[row, col].float().item() * scales[row, block_idx].item()
    return output


def ref_dynamic_fp8_quant(input_tensor: torch.Tensor,
                           block_size: int = 128) -> tuple:
    """Reference dynamic FP8 quantization: BF16/FP32 -> FP8 E4M3 with per-block scales.

    Args:
        input_tensor: [M, K] float/bfloat16 input
        block_size: elements per quantization block (128)

    Returns:
        (quantized [M, K] float32 in FP8 range, scales [M, num_blocks] float32)
    """
    FP8_MAX = 448.0
    inp = input_tensor.float()
    M, K = inp.shape
    num_blocks = (K + block_size - 1) // block_size

    scales = torch.zeros(M, num_blocks, dtype=torch.float32)
    output = torch.zeros(M, K, dtype=torch.float32)

    for row in range(M):
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, K)
            block = inp[row, start:end]
            amax = block.abs().max().item()
            scale = amax / FP8_MAX if amax > 0 else 1.0
            scales[row, b] = scale
            output[row, start:end] = torch.clamp(block / scale, -FP8_MAX, FP8_MAX)

    return output, scales


def ref_swiglu(gate: torch.Tensor, up: torch.Tensor,
               swiglu_limit: float = 0.0) -> torch.Tensor:
    """Reference SwiGLU: SiLU(gate) * up, with optional clamping.

    Clamp semantics follow llama.cpp LLM_ARCH_DEEPSEEK4
    (llama-graph.cpp build_ffn/build_moe_ffn, ggml_swiglu_split):
        gate = min(gate, limit)          # upper bound only
        up   = clamp(up, -limit, limit)  # two-sided
        out  = SiLU(gate) * up
    """
    g = gate.float()
    u = up.float()
    if swiglu_limit > 0.0:
        g = g.clamp(max=swiglu_limit)
        u = u.clamp(-swiglu_limit, swiglu_limit)
    return F.silu(g) * u


def ref_sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    """sqrt(softplus(x)) = sqrt(log(1 + exp(x)))."""
    return torch.sqrt(torch.nn.functional.softplus(x.float()))


def ref_topk_gating(logits: torch.Tensor, bias: torch.Tensor,
                     topk: int, n_group: int = 1, topk_group: int = 1,
                     routed_scaling_factor: float = 1.0,
                     renormalize: bool = False,
                     scoring_func: str = "sigmoid"):
    """Reference top-K gating with configurable scoring and optional grouped routing.

    Returns:
        (topk_weights [num_tokens, topk] float32,
         topk_indices [num_tokens, topk] int32)
    """
    num_tokens, num_experts = logits.shape
    if scoring_func == "sqrtsoftplus":
        scores = ref_sqrtsoftplus(logits)
    else:
        scores = torch.sigmoid(logits.float())
    sel_scores = scores + bias.float() if bias is not None else scores.clone()

    out_weights = torch.zeros(num_tokens, topk, dtype=torch.float32)
    out_indices = torch.zeros(num_tokens, topk, dtype=torch.int32)

    for t in range(num_tokens):
        if n_group <= 1:
            # Simple top-K on biased scores
            _, top_idx = sel_scores[t].topk(topk)
        else:
            # Grouped routing (DeepSeek V3.2 style)
            experts_per_group = num_experts // n_group

            # Phase 1: select top groups by sum of top-2 scores per group
            group_scores = torch.zeros(n_group)
            for g in range(n_group):
                grp = sel_scores[t, g * experts_per_group:(g + 1) * experts_per_group]
                k2 = min(2, len(grp))
                group_scores[g] = grp.topk(k2).values.sum()
            selected_groups = group_scores.topk(topk_group).indices

            # Phase 2: top-K from selected groups only
            mask = torch.full((num_experts,), float('-inf'))
            for g in selected_groups:
                start = g.item() * experts_per_group
                end = start + experts_per_group
                mask[start:end] = sel_scores[t, start:end]
            _, top_idx = mask.topk(topk)

        out_indices[t] = top_idx.int()
        # Routing weights use UNBIASED sigmoid scores
        out_weights[t] = scores[t, top_idx.long()]

        if renormalize:
            w_sum = out_weights[t].sum()
            if w_sum > 0:
                out_weights[t] *= routed_scaling_factor / w_sum

    return out_weights, out_indices


def ref_adaptive_gating(in_weights: torch.Tensor, in_indices: torch.Tensor,
                         threshold: float):
    """Reference adaptive gating: threshold-based expert pruning.

    Returns:
        (out_weights [num_tokens, topk] float32,
         out_indices [num_tokens, topk] int32,
         expert_counts [num_tokens] int32)
    """
    num_tokens, topk = in_weights.shape
    out_weights = torch.zeros_like(in_weights)
    out_indices = torch.full_like(in_indices, -1)
    expert_counts = torch.zeros(num_tokens, dtype=torch.int32)

    for t in range(num_tokens):
        # Sort by weight descending
        sorted_idx = torch.argsort(in_weights[t], descending=True)
        sorted_w = in_weights[t, sorted_idx]
        sorted_i = in_indices[t, sorted_idx]

        total = sorted_w.sum().item()
        target = threshold * total
        cumsum = 0.0
        count = topk
        for k in range(topk):
            cumsum += sorted_w[k].item()
            if cumsum >= target:
                count = k + 1
                break
        count = max(1, count)

        out_weights[t, :count] = sorted_w[:count]
        out_indices[t, :count] = sorted_i[:count]
        expert_counts[t] = count

    return out_weights, out_indices, expert_counts


def ref_grouped_gemm(activations: torch.Tensor, weights_list: list,
                      expert_offsets: torch.Tensor) -> torch.Tensor:
    """Reference grouped GEMM: per-expert torch.mm.

    Args:
        activations: [total_tokens, K] float32
        weights_list: list of [N, K] float32 weight matrices (one per expert)
        expert_offsets: [num_experts + 1] int tensor

    Returns:
        [total_tokens, N] float32 output
    """
    num_experts = len(weights_list)
    N = weights_list[0].shape[0]
    total_tokens = activations.shape[0]
    output = torch.zeros(total_tokens, N, dtype=torch.float32)

    for e in range(num_experts):
        start = expert_offsets[e].item()
        end = expert_offsets[e + 1].item()
        if start == end:
            continue
        x = activations[start:end].float()
        w = weights_list[e].float()
        output[start:end] = x @ w.T

    return output


def ref_expert_forward(tokens: torch.Tensor, expert_ids: torch.Tensor,
                        gate_weights: list, up_weights: list,
                        down_weights: list, gating_scores: torch.Tensor,
                        num_experts: int) -> torch.Tensor:
    """Reference full expert forward (BF16, per-expert torch.mm)."""
    B, hidden_dim = tokens.shape
    output = torch.zeros_like(tokens, dtype=torch.float32)

    for e in range(num_experts):
        mask = expert_ids == e
        if not mask.any():
            continue
        x = tokens[mask].float()  # [n_e, hidden_dim]
        gate_out = x @ gate_weights[e].float().T
        up_out = x @ up_weights[e].float().T
        activated = F.silu(gate_out) * up_out
        expert_out = activated @ down_weights[e].float().T
        output[mask] += expert_out * gating_scores[mask].unsqueeze(-1).float()

    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExpertReference(unittest.TestCase):
    """Reference tests establishing error budgets for expert kernels."""

    def test_swiglu_accuracy(self):
        """SwiGLU: SiLU(gate) * up must match manual computation exactly."""
        N, D = 64, 2048
        gate = torch.randn(N, D)
        up = torch.randn(N, D)

        result = ref_swiglu(gate, up)
        expected = torch.sigmoid(gate.float()) * gate.float() * up.float()

        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-7),
                        "SwiGLU must match sigmoid(x)*x*up")

    def test_permute_roundtrip(self):
        """Permute → unpermute must recover original order exactly."""
        B, D = 128, 512
        num_experts = 16
        tokens = torch.randn(B, D)
        expert_ids = torch.randint(0, num_experts, (B,))

        # Permute: group by expert
        sort_indices = torch.argsort(expert_ids, stable=True)
        permuted = tokens[sort_indices]

        # Unpermute: restore original order
        unpermute_indices = torch.argsort(sort_indices)
        recovered = permuted[unpermute_indices]

        self.assertTrue(torch.equal(tokens, recovered),
                        "Permute round-trip must be exact")

    def test_expert_forward_single_expert(self):
        """Single expert forward must match simple matmul chain."""
        B, hidden, inter = 32, 256, 64
        tokens = torch.randn(B, hidden)
        W_gate = torch.randn(inter, hidden)
        W_up = torch.randn(inter, hidden)
        W_down = torch.randn(hidden, inter)

        gate_out = tokens.float() @ W_gate.float().T
        up_out = tokens.float() @ W_up.float().T
        activated = F.silu(gate_out) * up_out
        expected = activated @ W_down.float().T

        result = ref_expert_forward(
            tokens,
            expert_ids=torch.zeros(B, dtype=torch.long),
            gate_weights=[W_gate],
            up_weights=[W_up],
            down_weights=[W_down],
            gating_scores=torch.ones(B),
            num_experts=1,
        )

        cosine = F.cosine_similarity(result.flatten(), expected.flatten(), dim=0)
        self.assertGreater(cosine, 0.9999,
                           f"Single expert forward cosine={cosine:.6f}")

    def test_nvfp4_dequant_accuracy(self):
        """NVFP4 dequant: verify LUT + scale produces correct values."""
        M, N = 16, 64
        group_size = 16
        fp4_lut = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ]

        weights_packed = torch.randint(0, 256, (M, N // 2), dtype=torch.uint8)
        scales = torch.randn(M, N // group_size).abs() + 0.01

        result = ref_nvfp4_dequant(weights_packed, scales, M, N, group_size)

        self.assertEqual(result.shape, (M, N))

        # Spot-check: first byte of first row
        byte_val = weights_packed[0, 0].item()
        lo_nib = byte_val & 0x0F
        hi_nib = (byte_val >> 4) & 0x0F
        scale_0 = scales[0, 0].item()  # col 0 and 1 are in group 0
        self.assertAlmostEqual(result[0, 0].item(), fp4_lut[lo_nib] * scale_0, places=5)
        self.assertAlmostEqual(result[0, 1].item(), fp4_lut[hi_nib] * scale_0, places=5)

        # Verify: all outputs are LUT entry * scale (finite values)
        self.assertTrue(torch.isfinite(result).all(), "All dequantized values should be finite")

    def test_fp8_dequant_accuracy(self):
        """FP8 dequant: value * scale must produce correct results."""
        M, N = 16, 256
        block_size = 128

        # Simulate FP8 values (small floats in representable range)
        data = torch.randn(M, N).clamp(-448, 448)
        num_blocks = (N + block_size - 1) // block_size
        scales = torch.randn(M, num_blocks).abs() + 0.01

        result = ref_fp8_dequant(data, scales, M, N, block_size)

        # Spot-check first element
        expected_00 = data[0, 0].item() * scales[0, 0].item()
        self.assertAlmostEqual(result[0, 0].item(), expected_00, places=5)

        # Check element in second block
        col = 130  # block index 1
        expected = data[0, col].item() * scales[0, 1].item()
        self.assertAlmostEqual(result[0, col].item(), expected, places=4)

    def test_dynamic_fp8_quant_roundtrip(self):
        """Quant -> dequant round-trip should have high cosine similarity."""
        M, K = 32, 512
        input_tensor = torch.randn(M, K)

        quantized, scales = ref_dynamic_fp8_quant(input_tensor)

        # Dequantize
        recovered = ref_fp8_dequant(quantized, scales, M, K, block_size=128)

        # Cosine similarity should be very high for random data
        cosine = F.cosine_similarity(
            input_tensor.float().flatten(),
            recovered.flatten(), dim=0)
        self.assertGreater(cosine.item(), 0.999,
                           f"FP8 quant roundtrip cosine={cosine:.6f}")

    def test_topk_gating_simple(self):
        """Simple top-K gating (n_group=1) must match torch.topk on sigmoid."""
        num_tokens, num_experts, topk = 16, 32, 4
        logits = torch.randn(num_tokens, num_experts)
        bias = torch.randn(num_experts) * 0.1

        weights, indices = ref_topk_gating(
            logits, bias, topk=topk, n_group=1, topk_group=1,
            routed_scaling_factor=1.0, renormalize=False)

        # Compare against manual: topk of (sigmoid + bias), weights from unbiased
        scores = torch.sigmoid(logits.float())
        sel_scores = scores + bias.float()
        _, manual_idx = sel_scores.topk(topk, dim=1)

        # Indices should match
        for t in range(num_tokens):
            self.assertTrue(
                set(indices[t].tolist()) == set(manual_idx[t].tolist()),
                f"Token {t}: indices mismatch")

        # Weights should be unbiased sigmoid at selected indices
        for t in range(num_tokens):
            for k in range(topk):
                expected_w = scores[t, indices[t, k].long()].item()
                self.assertAlmostEqual(weights[t, k].item(), expected_w, places=5)

    def test_topk_gating_grouped(self):
        """Grouped top-K gating (V3.2 style): n_group>1 selects groups first."""
        num_tokens = 8
        num_experts = 64  # 8 groups of 8
        n_group = 8
        topk_group = 4
        topk = 6

        logits = torch.randn(num_tokens, num_experts)
        bias = torch.zeros(num_experts)  # no bias for simplicity

        weights, indices = ref_topk_gating(
            logits, bias, topk=topk, n_group=n_group, topk_group=topk_group,
            routed_scaling_factor=2.5, renormalize=True)

        self.assertEqual(weights.shape, (num_tokens, topk))
        self.assertEqual(indices.shape, (num_tokens, topk))

        # Verify all selected experts come from at most topk_group groups
        experts_per_group = num_experts // n_group
        for t in range(num_tokens):
            groups_used = set()
            for k in range(topk):
                groups_used.add(indices[t, k].item() // experts_per_group)
            self.assertLessEqual(len(groups_used), topk_group,
                                 f"Token {t}: used {len(groups_used)} groups > {topk_group}")

        # Verify renormalized weights sum to routed_scaling_factor
        for t in range(num_tokens):
            w_sum = weights[t].sum().item()
            self.assertAlmostEqual(w_sum, 2.5, places=4,
                                   msg=f"Token {t}: weight sum={w_sum}, expected 2.5")

    def test_adaptive_gating_threshold(self):
        """Adaptive gating: threshold pruning with known inputs."""
        # 4 tokens, topk=4, threshold=0.9
        in_weights = torch.tensor([
            [0.50, 0.30, 0.15, 0.05],
            [0.90, 0.05, 0.03, 0.02],
            [0.25, 0.25, 0.25, 0.25],
            [0.40, 0.35, 0.20, 0.05],
        ], dtype=torch.float32)
        in_indices = torch.tensor([
            [10, 20, 30, 40],
            [5, 15, 25, 35],
            [1, 2, 3, 4],
            [100, 200, 300, 400],
        ], dtype=torch.int32)

        out_w, out_i, counts = ref_adaptive_gating(in_weights, in_indices, 0.9)

        # Token 0: cumsum 0.50, 0.80, 0.95 → 3 experts needed
        self.assertEqual(counts[0].item(), 3)
        # Token 1: cumsum 0.90 → 1 expert needed
        self.assertEqual(counts[1].item(), 1)
        # Token 2: cumsum 0.25, 0.50, 0.75, 1.0 → 4 experts (all needed for 0.9)
        self.assertEqual(counts[2].item(), 4)
        # Token 3: cumsum 0.40, 0.75, 0.95 → 3 experts
        self.assertEqual(counts[3].item(), 3)

        # Pruned slots should be sentinel
        self.assertEqual(out_i[0, 3].item(), -1)
        self.assertEqual(out_w[0, 3].item(), 0.0)
        for k in range(1, 4):
            self.assertEqual(out_i[1, k].item(), -1)

    def test_grouped_gemm_reference(self):
        """Grouped GEMM: per-expert matmul with varying token counts."""
        num_experts = 4
        K, N = 128, 64
        tokens_per_expert = [8, 0, 16, 4]  # expert 1 is empty
        total_tokens = sum(tokens_per_expert)

        activations = torch.randn(total_tokens, K)
        weights = [torch.randn(N, K) for _ in range(num_experts)]

        # Build expert_offsets
        offsets = [0]
        for t in tokens_per_expert:
            offsets.append(offsets[-1] + t)
        expert_offsets = torch.tensor(offsets, dtype=torch.int64)

        result = ref_grouped_gemm(activations, weights, expert_offsets)

        self.assertEqual(result.shape, (total_tokens, N))

        # Verify each expert's output matches torch.mm
        for e in range(num_experts):
            start = offsets[e]
            end = offsets[e + 1]
            if start == end:
                continue
            expected = activations[start:end].float() @ weights[e].float().T
            self.assertTrue(torch.allclose(result[start:end], expected, atol=1e-5),
                            f"Expert {e}: output mismatch")

        # Verify empty expert region is zero
        # Expert 1 has no tokens, so nothing to check

    def test_permute_with_topk_expansion(self):
        """Permute with topk>1: each token gets routed to multiple experts."""
        B, D = 16, 64
        num_experts = 8
        topk = 3
        tokens = torch.randn(B, D)
        expert_ids = torch.randint(0, num_experts, (B, topk))

        # Expand tokens: each token appears topk times
        expanded = tokens.unsqueeze(1).expand(-1, topk, -1).reshape(B * topk, D)

        # Flatten expert assignments
        flat_experts = expert_ids.reshape(-1)

        # Permute by expert
        sort_idx = torch.argsort(flat_experts, stable=True)
        permuted = expanded[sort_idx]

        # Unpermute
        unsort_idx = torch.argsort(sort_idx)
        recovered = permuted[unsort_idx]

        # Should exactly recover expanded tokens
        self.assertTrue(torch.equal(expanded, recovered),
                        "Topk-expanded permute roundtrip must be exact")


if __name__ == "__main__":
    unittest.main()

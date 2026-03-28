import pytest
import torch

from distillation.kd_loss_engine import KDLossEngine


def make_engine(**kwargs) -> KDLossEngine:
    defaults = dict(temperature=2.0, alpha=0.5, reasoning_weight=3.0)
    defaults.update(kwargs)
    return KDLossEngine(**defaults)


class TestKDLossEngineComputeKL:
    def test_output_shape(self):
        engine = make_engine()
        B, L, V, K = 2, 6, 1000, 64
        student_logits = torch.randn(B, L, V)
        teacher_indices = torch.randint(0, V, (B, L, K))
        teacher_logprobs = torch.randn(B, L, K)
        kl = engine.computeKLDivergence(student_logits, teacher_indices, teacher_logprobs)
        assert kl.shape == (B, L)

    def test_kl_non_negative(self):
        engine = make_engine()
        B, L, V, K = 1, 4, 200, 10
        student_logits = torch.randn(B, L, V)
        teacher_indices = torch.randint(0, V, (B, L, K))
        teacher_logprobs = torch.log_softmax(torch.randn(B, L, K), dim=-1)
        kl = engine.computeKLDivergence(student_logits, teacher_indices, teacher_logprobs)
        assert (kl >= -1e-4).all(), "KL divergence should be non-negative"


class TestKDLossEngineReasoningWeight:
    def test_reasoning_tokens_increase_loss(self):
        engine = make_engine(reasoning_weight=5.0)
        kl = torch.ones(2, 8)
        # All tokens are reasoning tokens
        mask_all = torch.ones(2, 8, dtype=torch.bool)
        # No reasoning tokens
        mask_none = torch.zeros(2, 8, dtype=torch.bool)
        loss_all = engine.applyReasoningWeight(kl, mask_all)
        loss_none = engine.applyReasoningWeight(kl, mask_none)
        assert loss_all.item() > loss_none.item()

    def test_no_reasoning_tokens_gives_mean_kl(self):
        engine = make_engine(reasoning_weight=3.0)
        kl = torch.tensor([[2.0, 4.0]])
        mask = torch.zeros(1, 2, dtype=torch.bool)
        loss = engine.applyReasoningWeight(kl, mask)
        assert abs(loss.item() - 3.0) < 1e-4  # mean of [2, 4] = 3


class TestKDLossEngineCompute:
    def test_compute_returns_all_keys(self):
        engine = make_engine()
        B, L, V, K = 1, 5, 300, 16
        student_logits = torch.randn(B, L, V, requires_grad=True)
        labels = torch.randint(0, V, (B, L))
        labels[0, :2] = -100  # mask prompt tokens
        teacher_indices = torch.randint(0, V, (B, L, K))
        teacher_logprobs = torch.log_softmax(torch.randn(B, L, K), dim=-1)
        reasoning_mask = torch.randint(0, 2, (B, L)).bool()

        out = engine.compute(
            student_logits, labels, teacher_indices, teacher_logprobs, reasoning_mask
        )
        assert "loss" in out
        assert "ce_loss" in out
        assert "kd_loss" in out

    def test_loss_is_scalar(self):
        engine = make_engine()
        B, L, V, K = 2, 4, 100, 8
        out = engine.compute(
            student_logits=torch.randn(B, L, V, requires_grad=True),
            labels=torch.randint(0, V, (B, L)),
            teacher_indices=torch.randint(0, V, (B, L, K)),
            teacher_logprobs=torch.log_softmax(torch.randn(B, L, K), dim=-1),
            reasoning_mask=torch.zeros(B, L, dtype=torch.bool),
        )
        assert out["loss"].shape == torch.Size([])

    def test_loss_is_differentiable(self):
        engine = make_engine()
        B, L, V, K = 1, 4, 50, 5
        student_logits = torch.randn(B, L, V, requires_grad=True)
        out = engine.compute(
            student_logits=student_logits,
            labels=torch.randint(0, V, (B, L)),
            teacher_indices=torch.randint(0, V, (B, L, K)),
            teacher_logprobs=torch.log_softmax(torch.randn(B, L, K), dim=-1),
            reasoning_mask=torch.zeros(B, L, dtype=torch.bool),
        )
        out["loss"].backward()
        assert student_logits.grad is not None

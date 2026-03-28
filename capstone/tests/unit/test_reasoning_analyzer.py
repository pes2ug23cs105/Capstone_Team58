import pytest
import torch

from reasoning.entropy import compute_token_entropy
from reasoning.mask_generator import generate_reasoning_mask
from reasoning.reasoning_analyzer import ReasoningAnalyzer


class TestComputeTokenEntropy:
    def test_uniform_distribution_has_max_entropy(self):
        vocab_size = 100
        # All logits equal → uniform distribution → max entropy
        logits = torch.zeros(5, vocab_size)
        entropy = compute_token_entropy(logits)
        expected = torch.log(torch.tensor(float(vocab_size)))
        assert entropy.shape == (5,)
        assert torch.allclose(entropy, expected.expand(5), atol=1e-4)

    def test_peaked_distribution_has_low_entropy(self):
        vocab_size = 100
        # One logit very large → near-zero entropy
        logits = torch.full((3, vocab_size), -100.0)
        logits[:, 0] = 100.0
        entropy = compute_token_entropy(logits)
        assert (entropy < 0.01).all()

    def test_output_shape(self):
        logits = torch.randn(10, 32000)
        entropy = compute_token_entropy(logits)
        assert entropy.shape == (10,)

    def test_non_negative(self):
        logits = torch.randn(8, 50)
        entropy = compute_token_entropy(logits)
        assert (entropy >= 0).all()


class TestGenerateReasoningMask:
    def test_all_above_threshold(self):
        entropy = torch.tensor([2.0, 3.0, 2.5])
        mask = generate_reasoning_mask(entropy, threshold=1.5)
        assert mask.all()

    def test_all_below_threshold(self):
        entropy = torch.tensor([0.1, 0.2, 0.5])
        mask = generate_reasoning_mask(entropy, threshold=1.5)
        assert not mask.any()

    def test_mixed(self):
        entropy = torch.tensor([0.5, 2.0, 1.5, 0.3, 3.0])
        mask = generate_reasoning_mask(entropy, threshold=1.5)
        assert mask.tolist() == [False, True, True, False, True]

    def test_threshold_boundary(self):
        entropy = torch.tensor([1.5])
        mask = generate_reasoning_mask(entropy, threshold=1.5)
        assert mask[0].item() is True  # >= is inclusive


class TestReasoningAnalyzer:
    def setup_method(self):
        self.analyzer = ReasoningAnalyzer(entropy_threshold=1.5)

    def test_compute_entropy_returns_correct_shape(self):
        logits = torch.randn(7, 200)
        entropy = self.analyzer.computeEntropy(logits)
        assert entropy.shape == (7,)

    def test_detect_decision_points_returns_indices(self):
        entropy = torch.tensor([0.1, 2.0, 0.5, 3.0])
        indices = self.analyzer.detectDecisionPoints(entropy)
        assert indices.tolist() == [1, 3]

    def test_detect_decision_points_empty(self):
        entropy = torch.tensor([0.1, 0.2])
        indices = self.analyzer.detectDecisionPoints(entropy)
        assert len(indices) == 0

    def test_generate_mask_consistency(self):
        logits = torch.randn(10, 500)
        entropy = self.analyzer.computeEntropy(logits)
        mask = self.analyzer.generateMask(entropy)
        dp = self.analyzer.detectDecisionPoints(entropy)
        # Every True position in mask should be in decision points
        assert set(torch.where(mask)[0].tolist()) == set(dp.tolist())

"""
Integration test for the teacher inference pipeline.

Uses tiny synthetic models (no GPU required) to verify the full
data flow: sample → teacher → logits → entropy → mask → DistillationRecord.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from data.schemas import DistillationRecord
from reasoning.reasoning_analyzer import ReasoningAnalyzer
from teacher.logit_extractor import LogitExtractor


class TestTeacherPipelineIntegration:
    """
    Tests the teacher pipeline components in combination without loading
    real model weights (all model calls are mocked).
    """

    def _make_fake_logits(self, seq_len: int = 10, vocab_size: int = 500) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(seq_len, vocab_size)

    def test_extractor_then_analyzer_produces_valid_record(self):
        logits = self._make_fake_logits(seq_len=10, vocab_size=500)

        extractor = LogitExtractor(top_k=8)
        topk_indices, topk_logprobs = extractor.extract(logits)

        analyzer = ReasoningAnalyzer(entropy_threshold=1.0)
        entropy = analyzer.computeEntropy(logits)
        mask = analyzer.generateMask(entropy)

        record = DistillationRecord(
            sample_id="test-001",
            source="mathvista",
            question="What is 2+2?",
            answer="4",
            rationale="2 + 2 = 4",
            topk_indices=topk_indices,
            topk_logits=topk_logprobs,
            entropy_scores=entropy.tolist(),
            reasoning_mask=mask.tolist(),
        )

        assert len(record.topk_indices) == 10
        assert len(record.topk_indices[0]) == 8
        assert len(record.reasoning_mask) == 10
        assert record.entropy_scores_np.shape == (10,)

    def test_top_k_indices_are_valid_vocab_indices(self):
        vocab_size = 500
        logits = self._make_fake_logits(vocab_size=vocab_size)
        extractor = LogitExtractor(top_k=16)
        topk_indices, _ = extractor.extract(logits)
        for pos_indices in topk_indices:
            for idx in pos_indices:
                assert 0 <= idx < vocab_size

    def test_topk_logprobs_are_log_probs(self):
        logits = self._make_fake_logits()
        extractor = LogitExtractor(top_k=5)
        _, topk_logprobs = extractor.extract(logits)
        for pos_lp in topk_logprobs:
            for lp in pos_lp:
                assert lp <= 0.0, "log-probs must be non-positive"

    def test_jsonl_serialisation_roundtrip(self):
        logits = self._make_fake_logits(seq_len=5, vocab_size=100)
        extractor = LogitExtractor(top_k=4)
        analyzer = ReasoningAnalyzer(entropy_threshold=0.5)

        topk_indices, topk_logprobs = extractor.extract(logits)
        entropy = analyzer.computeEntropy(logits)
        mask = analyzer.generateMask(entropy)

        original = DistillationRecord(
            sample_id="abc",
            source="vsr",
            question="Is the cat on the table?",
            answer="true",
            rationale=None,
            topk_indices=topk_indices,
            topk_logits=topk_logprobs,
            entropy_scores=entropy.tolist(),
            reasoning_mask=mask.tolist(),
        )
        line = original.to_jsonl_line()
        restored = DistillationRecord.from_jsonl_line(line)

        assert restored.sample_id == original.sample_id
        assert restored.answer == original.answer
        assert restored.reasoning_mask == original.reasoning_mask
        assert restored.topk_indices == original.topk_indices

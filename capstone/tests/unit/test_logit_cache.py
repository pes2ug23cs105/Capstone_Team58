import pytest
import json
import tempfile
from pathlib import Path

from data.schemas import DistillationRecord
from distillation.logit_cache import LogitCache


def _make_record(i: int) -> DistillationRecord:
    return DistillationRecord(
        sample_id=f"s{i}",
        source="mathvista",
        question=f"Question {i}",
        answer=f"Answer {i}",
        rationale=f"Reasoning {i}",
        topk_indices=[[j for j in range(4)] for _ in range(5)],
        topk_logits=[[-1.0, -2.0, -3.0, -4.0] for _ in range(5)],
        entropy_scores=[0.5, 1.0, 2.0, 0.8, 1.5],
        reasoning_mask=[False, False, True, False, True],
    )


class TestLogitCache:
    def test_write_and_read_roundtrip(self, tmp_path):
        cache = LogitCache(output_dir=str(tmp_path), chunk_size=10)
        records = [_make_record(i) for i in range(5)]
        cache.write(iter(records))
        loaded = list(cache.read())
        assert len(loaded) == 5
        assert loaded[0].sample_id == "s0"
        assert loaded[4].answer == "Answer 4"

    def test_chunking_creates_multiple_files(self, tmp_path):
        cache = LogitCache(output_dir=str(tmp_path), chunk_size=3)
        records = [_make_record(i) for i in range(7)]
        cache.write(iter(records))
        files = sorted(tmp_path.glob("topk_logits_*.jsonl"))
        assert len(files) == 3  # 3 + 3 + 1

    def test_count_matches_write(self, tmp_path):
        cache = LogitCache(output_dir=str(tmp_path), chunk_size=5)
        records = [_make_record(i) for i in range(12)]
        n = cache.write(iter(records))
        assert n == 12
        assert cache.count() == 12

    def test_read_empty_raises(self, tmp_path):
        cache = LogitCache(output_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            list(cache.read())

    def test_reasoning_mask_preserved(self, tmp_path):
        cache = LogitCache(output_dir=str(tmp_path))
        rec = _make_record(0)
        cache.write(iter([rec]))
        loaded = list(cache.read())[0]
        assert loaded.reasoning_mask == rec.reasoning_mask

    def test_distillation_record_numpy_properties(self, tmp_path):
        import numpy as np
        rec = _make_record(0)
        assert rec.entropy_scores_np.dtype == np.float32
        assert rec.reasoning_mask_np.dtype == bool
        assert rec.topk_indices_np.shape == (5, 4)

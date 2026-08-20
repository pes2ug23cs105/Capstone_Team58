from __future__ import annotations

import pipelines.phase1_teacher_distill as phase1
from data.schemas import DistillationRecord


def _make_record(sample_id: str) -> DistillationRecord:
    return DistillationRecord(
        sample_id=sample_id,
        source="mathvista",
        question="q",
        answer="a",
        rationale=None,
        topk_indices=[[1, 2]],
        topk_logits=[[-0.1, -0.2]],
        entropy_scores=[1.0],
        reasoning_mask=[True],
    )


class TestPhase1Resume:
    def test_resume_skips_already_cached_ids(self, monkeypatch):
        class FakeManager:
            def iter_samples(self):
                return iter(
                    [
                        {"id": "done-1", "source": "mathvista", "question": "q1", "answer": "a1", "image": None},
                        {"id": "new-1", "source": "mathvista", "question": "q2", "answer": "a2", "image": None},
                        {"id": "new-2", "source": "vsr", "question": "q3", "answer": "a3", "image": None},
                    ]
                )

        class FakeDatasetManager:
            @classmethod
            def from_config(cls, _train_config):
                return FakeManager()

        class FakePipeline:
            last_seen_ids: list[str] = []

            def __init__(self, config_path=None, disengagement_mode=None, max_new_tokens_override=None):
                self.config_path = config_path

            def run(self, samples, total=None):
                FakePipeline.last_seen_ids = [s["id"] for s in samples]
                for sample_id in FakePipeline.last_seen_ids:
                    yield _make_record(sample_id)

        class FakeCache:
            written_ids: list[str] = []

            @classmethod
            def from_config(cls, _train_config):
                return cls()

            def cached_ids(self):
                return {"done-1"}

            def write(self, records):
                FakeCache.written_ids = [r.sample_id for r in records]
                return len(FakeCache.written_ids)

        monkeypatch.setattr(phase1, "DatasetManager", FakeDatasetManager)
        monkeypatch.setattr(phase1, "TeacherInferencePipeline", FakePipeline)
        monkeypatch.setattr(phase1, "LogitCache", FakeCache)

        phase1.run_phase1(model_config="config/model_config.yaml", train_config="config/training_config.yaml")

        assert FakePipeline.last_seen_ids == ["new-1", "new-2"]
        assert FakeCache.written_ids == ["new-1", "new-2"]

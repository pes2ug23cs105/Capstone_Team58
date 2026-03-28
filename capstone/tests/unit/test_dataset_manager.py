import pytest
from unittest.mock import MagicMock, patch

from data.adapters.base_adapter import BaseDatasetAdapter
from data.pipeline.dataset_manager import DatasetManager
from data.pipeline.sampler import WeightedMultiDatasetSampler


# ------------------------------------------------------------------ #
#  Minimal concrete adapter for testing                               #
# ------------------------------------------------------------------ #

class _StubAdapter(BaseDatasetAdapter):
    def __init__(self, records: list[dict]):
        self._records = records

    def load(self):
        yield from self._records

    def normalize(self, sample: dict) -> dict:
        return {**sample, "source": "stub"}

    def prefilter(self, sample: dict) -> bool:
        return bool(sample.get("question"))


class TestDatasetManagerIterSamples:
    def test_iterates_all_adapters(self):
        a1 = _StubAdapter([
            {"id": "1", "question": "Q1", "answer": "A1", "image": None, "rationale": None},
            {"id": "2", "question": "Q2", "answer": "A2", "image": None, "rationale": None},
        ])
        a2 = _StubAdapter([
            {"id": "3", "question": "Q3", "answer": "A3", "image": None, "rationale": None},
        ])
        manager = DatasetManager([a1, a2])
        samples = list(manager.iter_samples())
        assert len(samples) == 3

    def test_prefilter_removes_empty_questions(self):
        adapter = _StubAdapter([
            {"id": "1", "question": "", "answer": "A", "image": None, "rationale": None},
            {"id": "2", "question": "Q", "answer": "A", "image": None, "rationale": None},
        ])
        manager = DatasetManager([adapter])
        samples = list(manager.iter_samples())
        assert len(samples) == 1
        assert samples[0]["id"] == "2"

    def test_source_tag_is_set(self):
        adapter = _StubAdapter([
            {"id": "1", "question": "Q", "answer": "A", "image": None, "rationale": None},
        ])
        manager = DatasetManager([adapter])
        sample = next(manager.iter_samples())
        assert sample["source"] == "stub"


class TestWeightedMultiDatasetSampler:
    def test_only_produces_registered_names(self):
        weights = {"mathvista": 0.5, "vsr": 0.3, "egoschema": 0.2}
        sampler = WeightedMultiDatasetSampler(weights, seed=0)
        names = {next(sampler) for _ in range(200)}
        assert names <= set(weights.keys())

    def test_weights_are_respected_approximately(self):
        weights = {"a": 0.9, "b": 0.1}
        sampler = WeightedMultiDatasetSampler(weights, seed=42)
        counts = {"a": 0, "b": 0}
        for _ in range(1000):
            counts[next(sampler)] += 1
        # "a" should appear ~90% of the time (allow ±10%)
        assert counts["a"] > 750

    def test_iter_is_infinite(self):
        sampler = WeightedMultiDatasetSampler({"x": 1.0}, seed=0)
        it = iter(sampler)
        for _ in range(500):
            assert next(it) == "x"

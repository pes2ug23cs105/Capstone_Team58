from __future__ import annotations

from PIL import Image

from data.adapters.egoschema_adapter import EgoSchemaAdapter
from data.adapters.mathvista_adapter import MathVistaAdapter
from data.adapters.vsr_adapter import VSRAdapter


def _small_image() -> Image.Image:
    return Image.new("RGB", (16, 16), color="white")


def _valid_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color="white")


class TestAdapterPrefilter:
    def test_question_too_long_is_filtered_mathvista(self):
        adapter = MathVistaAdapter()
        sample = {
            "id": "m1",
            "question": "q" * (adapter.MAX_QUESTION_LENGTH + 1),
            "answer": "42",
            "image": None,
            "rationale": None,
            "source": "mathvista",
        }
        assert adapter.prefilter(sample) is False

    def test_question_too_long_is_filtered_egoschema(self):
        adapter = EgoSchemaAdapter()
        sample = {
            "id": "e1",
            "question": "q" * (adapter.MAX_QUESTION_LENGTH + 1),
            "answer": "option",
            "image": None,
            "rationale": None,
            "source": "egoschema",
        }
        assert adapter.prefilter(sample) is False

    def test_question_too_long_is_filtered_vsr(self):
        adapter = VSRAdapter()
        sample = {
            "id": "v1",
            "question": "q" * (adapter.MAX_QUESTION_LENGTH + 1),
            "answer": "true",
            "image": _valid_image(),
            "rationale": None,
            "source": "vsr",
        }
        assert adapter.prefilter(sample) is False

    def test_image_below_min_dim_is_filtered_mathvista(self):
        adapter = MathVistaAdapter()
        sample = {
            "id": "m2",
            "question": "valid question text",
            "answer": "42",
            "image": _small_image(),
            "rationale": None,
            "source": "mathvista",
        }
        assert adapter.prefilter(sample) is False

    def test_image_below_min_dim_is_filtered_egoschema(self):
        adapter = EgoSchemaAdapter()
        sample = {
            "id": "e2",
            "question": "valid question text",
            "answer": "option",
            "image": _small_image(),
            "rationale": None,
            "source": "egoschema",
        }
        assert adapter.prefilter(sample) is False

    def test_image_below_min_dim_is_filtered_vsr(self):
        adapter = VSRAdapter()
        sample = {
            "id": "v2",
            "question": "valid question text",
            "answer": "true",
            "image": _small_image(),
            "rationale": None,
            "source": "vsr",
        }
        assert adapter.prefilter(sample) is False


class TestAdapterNormalizeText:
    def test_mathvista_unicode_text_is_normalized(self):
        adapter = MathVistaAdapter()
        raw = {
            "pid": "m3",
            "question": "What\u00a0 is\n\n2 + 2?",
            "answer": " 4\n",
            "image": None,
        }

        normalized = adapter.normalize(raw)

        assert normalized["question"] == "What is 2 + 2?"
        assert normalized["answer"] == "4"

    def test_egoschema_unicode_text_is_normalized(self):
        adapter = EgoSchemaAdapter()
        raw = {
            "q_uid": "e3",
            "question": "Which\u00a0option\n is correct?",
            "option 0": "A",
            "option 1": "B\n\nvalue",
            "option 2": "",
            "option 3": "",
            "option 4": "",
            "answer": 1,
        }

        normalized = adapter.normalize(raw)

        assert "\u00a0" not in normalized["question"]
        assert "  " not in normalized["question"]
        assert normalized["answer"] == "B value"

    def test_vsr_unicode_text_is_normalized(self):
        adapter = VSRAdapter()
        raw = {
            "id": "v3",
            "caption": "cat\u00a0\n\nunder table",
            "label": 1,
            "image": _valid_image(),
        }

        normalized = adapter.normalize(raw)

        assert "\u00a0" not in normalized["question"]
        assert "  " not in normalized["question"]
        assert '"cat under table"' in normalized["question"]
        assert normalized["answer"] == "true"

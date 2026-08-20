from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from data.schemas import DistillationRecord
from distillation.logit_cache import LogitCache


class DistillationDataset(Dataset):
    """
    PyTorch Dataset that wraps the JSONL logit cache for student training.

    Each item returns a dict with:
        input_ids        : (seq_len,)       tokenised prompt
        labels           : (seq_len,)       shifted labels (-100 at prompt positions)
        teacher_indices  : (seq_len, K)     Top-K teacher token indices
        teacher_logprobs : (seq_len, K)     corresponding log-probs
        reasoning_mask   : (seq_len,)       bool — True = reasoning token
    """

    def __init__(
        self,
        cache: LogitCache,
        processor: AutoProcessor,
        max_seq_length: int = 2048,
        allowed_sources: set[str] | None = None,
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        # Load all records into memory; for large caches use a streaming variant
        records = list(cache.read())
        if allowed_sources is not None:
            records = [r for r in records if r.source in allowed_sources]
        self._records: list[DistillationRecord] = records
        self.sources: list[str] = [r.source for r in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self._records[idx]
        L = self.max_seq_length
        topk_indices_raw = self._normalize_topk_shape(rec.topk_indices, "topk_indices")
        topk_logits_raw = self._normalize_topk_shape(rec.topk_logits, "topk_logits")
        K = len(topk_indices_raw[0]) if topk_indices_raw else 64

        if len(topk_indices_raw) != len(topk_logits_raw):
            raise ValueError(
                "topk_indices/topk_logits length mismatch: "
                f"{len(topk_indices_raw)} != {len(topk_logits_raw)}"
            )
        if topk_indices_raw and topk_logits_raw:
            if len(topk_indices_raw[0]) != len(topk_logits_raw[0]):
                raise ValueError(
                    "topk_indices/topk_logits K mismatch: "
                    f"{len(topk_indices_raw[0])} != {len(topk_logits_raw[0])}"
                )

        # Tokenise the question (prompt only — labels mask these positions)
        enc = self.processor.tokenizer(
            rec.question,
            truncation=True,
            max_length=L,
            return_tensors="pt",
        )
        prompt_ids = enc["input_ids"][0]   # (prompt_len,)

        # Tokenise the full sequence (prompt + answer)
        full_text = rec.question + " " + rec.answer
        full_enc = self.processor.tokenizer(
            full_text,
            truncation=True,
            max_length=L,
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"][0]   # (seq_len,)
        seq_len = input_ids.size(0)
        prompt_len = prompt_ids.size(0)

        # Labels: -100 for prompt tokens, real ids for answer tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # Pad or truncate teacher tensors to seq_len
        teacher_indices = self._pad_or_truncate_2d(topk_indices_raw, seq_len, K, 0)
        teacher_logprobs = self._pad_or_truncate_2d(topk_logits_raw, seq_len, K, -1e9)
        entropy = self._pad_or_truncate_1d(rec.entropy_scores, seq_len, 0.0)
        reasoning_mask = self._pad_or_truncate_1d(rec.reasoning_mask, seq_len, False)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "teacher_indices": torch.tensor(teacher_indices, dtype=torch.long),
            "teacher_logprobs": torch.tensor(teacher_logprobs, dtype=torch.float32),
            "reasoning_mask": torch.tensor(reasoning_mask, dtype=torch.bool),
        }

    # ------------------------------------------------------------------ #

    @staticmethod
    def _pad_or_truncate_2d(
        data: list[list], target_len: int, k: int, pad_val
    ) -> list[list]:
        data = data[:target_len]
        while len(data) < target_len:
            data.append([pad_val] * k)
        return data

    @staticmethod
    def _pad_or_truncate_1d(data: list, target_len: int, pad_val) -> list:
        data = list(data[:target_len])
        while len(data) < target_len:
            data.append(pad_val)
        return data

    @staticmethod
    def _normalize_topk_shape(data: list, field_name: str) -> list[list]:
        """
        Accept both (L, K) and legacy (L, 1, K); normalize to (L, K).
        """
        if not data:
            return []
        if not isinstance(data, list):
            raise ValueError(f"{field_name} must be a list, got {type(data)}")

        first = data[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            collapsed: list[list] = []
            for row in data:
                if len(row) != 1:
                    raise ValueError(
                        f"{field_name} legacy 3D shape requires middle dim size 1, got {len(row)}"
                    )
                collapsed.append(row[0])
            data = collapsed

        if not data or not isinstance(data[0], list):
            raise ValueError(f"{field_name} must be 2D after normalization")
        if data[0] and isinstance(data[0][0], list):
            raise ValueError(f"{field_name} still has rank > 2 after normalization")
        return data

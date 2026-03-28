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
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        # Load all records into memory; for large caches use a streaming variant
        self._records: list[DistillationRecord] = list(cache.read())

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self._records[idx]
        L = self.max_seq_length
        K = len(rec.topk_indices[0]) if rec.topk_indices else 64

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
        teacher_indices = self._pad_or_truncate_2d(rec.topk_indices, seq_len, K, 0)
        teacher_logprobs = self._pad_or_truncate_2d(rec.topk_logits, seq_len, K, -1e9)
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

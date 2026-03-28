from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import yaml

from data.schemas import DistillationRecord

logger = logging.getLogger(__name__)


class LogitCache:
    """
    Handles reading and writing of DistillationRecords to JSONL files.

    Each JSONL file holds `chunk_size` records.  Files are named:
        {output_dir}/{prefix}_0000.jsonl
        {output_dir}/{prefix}_0001.jsonl
        ...

    This chunked layout keeps individual files manageable and allows
    parallel reads during training.
    """

    def __init__(
        self,
        output_dir: str = "outputs/logits",
        file_prefix: str = "topk_logits",
        chunk_size: int = 1000,
    ):
        self.output_dir = Path(output_dir)
        self.file_prefix = file_prefix
        self.chunk_size = chunk_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str = "config/training_config.yaml") -> "LogitCache":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["logit_cache"]
        return cls(
            output_dir=cfg["output_dir"],
            file_prefix=cfg["file_prefix"],
            chunk_size=cfg["chunk_size"],
        )

    # ------------------------------------------------------------------ #
    #  Writing                                                             #
    # ------------------------------------------------------------------ #

    def write(self, records: Iterator[DistillationRecord]) -> int:
        """
        Consume an iterator of DistillationRecords and write them to chunked
        JSONL files.

        Returns:
            total number of records written.
        """
        chunk_idx = self._next_chunk_index()
        buffer: list[str] = []
        total = 0

        for record in records:
            buffer.append(record.to_jsonl_line())
            total += 1

            if len(buffer) >= self.chunk_size:
                self._flush(buffer, chunk_idx)
                buffer.clear()
                chunk_idx += 1

        if buffer:
            self._flush(buffer, chunk_idx)

        files_written = 0
        if total > 0:
            files_written = (total - 1) // self.chunk_size + 1
        logger.info("LogitCache: wrote %d records across %d files.", total, files_written)
        return total

    def _flush(self, lines: list[str], chunk_idx: int) -> None:
        path = self.output_dir / f"{self.file_prefix}_{chunk_idx:04d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _next_chunk_index(self) -> int:
        files = sorted(self.output_dir.glob(f"{self.file_prefix}_*.jsonl"))
        if not files:
            return 0

        indices: list[int] = []
        for path in files:
            suffix = path.stem.replace(f"{self.file_prefix}_", "")
            try:
                indices.append(int(suffix))
            except ValueError:
                continue
        if not indices:
            return 0
        return max(indices) + 1

    # ------------------------------------------------------------------ #
    #  Reading                                                             #
    # ------------------------------------------------------------------ #

    def read(self) -> Iterator[DistillationRecord]:
        """Yield DistillationRecords from all cached JSONL files in order."""
        files = sorted(self.output_dir.glob(f"{self.file_prefix}_*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"No cache files found in '{self.output_dir}' "
                f"matching prefix '{self.file_prefix}'."
            )
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield DistillationRecord.from_jsonl_line(line)
                        except Exception as e:
                            logger.warning("Skipping malformed record in %s: %s", path, e)

    def count(self) -> int:
        """Count total cached records without loading them into memory."""
        total = 0
        for path in sorted(self.output_dir.glob(f"{self.file_prefix}_*.jsonl")):
            with open(path, encoding="utf-8") as f:
                total += sum(1 for line in f if line.strip())
        return total

    def cached_ids(self) -> set[str]:
        """Return set of sample_ids already written to cache."""
        ids: set[str] = set()
        files = sorted(self.output_dir.glob(f"{self.file_prefix}_*.jsonl"))
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        sample_id = payload.get("sample_id")
                        if sample_id is not None:
                            ids.add(str(sample_id))
                    except Exception:
                        continue
        return ids

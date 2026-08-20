"""
Phase 1: Offline Teacher Distillation
======================================
Run the Qwen2.5-VL-32B teacher over all datasets, extract Top-K sparse logits,
compute token entropy + reasoning masks, and write DistillationRecords to the
JSONL logit cache on disk.

Usage:
    python -m pipelines.phase1_teacher_distill
    python -m pipelines.phase1_teacher_distill --model-config config/model_config.yaml
                                                --train-config config/training_config.yaml
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

from data.pipeline.dataset_manager import DatasetManager
from distillation.logit_cache import LogitCache
from teacher.inference_pipeline import TeacherInferencePipeline


class _SourceFilter(logging.Filter):
    def __init__(self, source: str):
        super().__init__()
        self._needle = f"source={source}"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._needle in record.getMessage()


_REPORTS_DIR = Path("outputs/reports")
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
_PHASE1_LOG_PATH = _REPORTS_DIR / "phase1_tstar.log"
_MATHVISTA_LOG_PATH = _REPORTS_DIR / "phase1_tstar_mathvista.log"
_EGOSCHEMA_LOG_PATH = _REPORTS_DIR / "phase1_tstar_egoschema.log"

_stream_handler = logging.StreamHandler()
_all_handler = logging.FileHandler(_PHASE1_LOG_PATH, mode="a", encoding="utf-8")

_mathvista_handler = logging.FileHandler(_MATHVISTA_LOG_PATH, mode="a", encoding="utf-8")
_mathvista_handler.addFilter(_SourceFilter("mathvista"))

_egoschema_handler = logging.FileHandler(_EGOSCHEMA_LOG_PATH, mode="a", encoding="utf-8")
_egoschema_handler.addFilter(_SourceFilter("egoschema"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        _stream_handler,
        _all_handler,
        _mathvista_handler,
        _egoschema_handler,
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def run_phase1(
    model_config: str = "config/model_config.yaml",
    train_config: str = "config/training_config.yaml",
    max_samples: int | None = None,
    max_new_tokens: int | None = None,
    disengagement_mode: str = "approx",
) -> None:
    logger.info("=== Phase 1: Offline Teacher Distillation ===")

    # 1. Load and iterate all datasets
    logger.info("Loading datasets...")
    manager = DatasetManager.from_config(train_config)

    dataset_names = [
        getattr(adapter, "SOURCE_TAG", adapter.__class__.__name__.lower())
        for adapter in getattr(manager, "_adapters", [])
    ]
    print("Loaded dataset:", ", ".join(dataset_names) if dataset_names else "unknown")

    # 2. Build teacher pipeline (loads the 32B model)
    logger.info("Initialising teacher model (this may take several minutes)...")
    pipeline = TeacherInferencePipeline(
        config_path=model_config,
        disengagement_mode=disengagement_mode,
        max_new_tokens_override=max_new_tokens,
    )

    # 3. Build logit cache writer
    cache = LogitCache.from_config(train_config)

    # 4. Resume support: skip samples already cached on disk
    done_ids = cache.cached_ids()
    logger.info("Resuming: %d samples already cached, skipping them.", len(done_ids))

    def _skip_done(samples):
        seen_in_run: set[str] = set()
        for sample in samples:
            sample_id = str(sample.get("id", "")).strip()
            if not sample_id:
                raise ValueError("Missing sample id in phase1 input stream.")
            if sample_id in seen_in_run:
                raise ValueError(f"Duplicate sample_id in current phase1 run: {sample_id}")
            seen_in_run.add(sample_id)
            if sample_id not in done_ids:
                yield sample

    sample_iter = _skip_done(manager.iter_samples())
    if max_samples is not None:
        sample_iter = itertools.islice(sample_iter, max_samples)

    # 5. Run inference and write cache
    logger.info("Running teacher inference + writing logit cache...")
    total_written = cache.write(pipeline.run(sample_iter, total=max_samples))

    logger.info("Phase 1 complete. %d records written to cache.", total_written)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--train-config", default="config/training_config.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--disengagement-mode", default="approx")
    args = parser.parse_args()
    run_phase1(
        args.model_config,
        args.train_config,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        disengagement_mode=args.disengagement_mode,
    )

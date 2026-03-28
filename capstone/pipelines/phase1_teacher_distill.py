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
import logging

from data.pipeline.dataset_manager import DatasetManager
from distillation.logit_cache import LogitCache
from teacher.inference_pipeline import TeacherInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_phase1(
    model_config: str = "config/model_config.yaml",
    train_config: str = "config/training_config.yaml",
) -> None:
    logger.info("=== Phase 1: Offline Teacher Distillation ===")

    # 1. Load and iterate all datasets
    logger.info("Loading datasets...")
    manager = DatasetManager.from_config(train_config)

    # 2. Build teacher pipeline (loads the 32B model)
    logger.info("Initialising teacher model (this may take several minutes)...")
    pipeline = TeacherInferencePipeline(config_path=model_config)

    # 3. Build logit cache writer
    cache = LogitCache.from_config(train_config)

    # 4. Resume support: skip samples already cached on disk
    done_ids = cache.cached_ids()
    logger.info("Resuming: %d samples already cached, skipping them.", len(done_ids))

    def _skip_done(samples):
        for sample in samples:
            if sample["id"] not in done_ids:
                yield sample

    # 5. Run inference and write cache
    logger.info("Running teacher inference + writing logit cache...")
    total_written = cache.write(pipeline.run(_skip_done(manager.iter_samples())))

    logger.info("Phase 1 complete. %d records written to cache.", total_written)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--train-config", default="config/training_config.yaml")
    args = parser.parse_args()
    run_phase1(args.model_config, args.train_config)

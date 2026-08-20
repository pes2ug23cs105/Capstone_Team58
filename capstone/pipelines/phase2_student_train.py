"""
Phase 2: Focused Student Training
====================================
Load the cached logit files produced in Phase 1, fine-tune Qwen2.5-VL-3B/7B
with LoRA using the reasoning-weighted KD loss, and save adapter checkpoints.

Usage:
    python -m pipelines.phase2_student_train
    python -m pipelines.phase2_student_train --model-config config/model_config.yaml
                                              --train-config config/training_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from distillation.distillation_dataset import DistillationDataset
from distillation.kd_loss_engine import KDLossEngine
from distillation.logit_cache import LogitCache
from student.lora_adapter import LoRAAdapter
from student.student_model import StudentModel
from student.trainer import DistillationTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_phase2(
    model_config: str = "config/model_config.yaml",
    train_config: str = "config/training_config.yaml",
    max_steps: int | None = None,
) -> None:
    logger.info("=== Phase 2: Focused Student Training ===")

    with open(train_config) as f:
        full_cfg = yaml.safe_load(f)
    tcfg = full_cfg["training"]
    dataset_cfg = full_cfg.get("datasets", {})

    # 1. Load cached teacher logits
    logger.info("Loading logit cache...")
    cache = LogitCache.from_config(train_config)
    num_records = cache.count()
    logger.info("Cache contains %d distillation records.", num_records)

    # 2. Build student model
    logger.info("Initialising student model...")
    student = StudentModel(config_path=model_config)

    # 3. Build dataset and dataloader
    # Restrict to configured training sources only — the cache may contain
    # records from sources no longer approved for training (e.g. MathVista,
    # whose license prohibits training use; it stays eval-only).
    allowed_sources = set(dataset_cfg.keys())
    dataset = DistillationDataset(
        cache=cache,
        processor=student.processor,
        max_seq_length=tcfg["max_seq_length"],
        allowed_sources=allowed_sources,
    )

    source_counts = Counter(dataset.sources)
    logger.info("Dataset size: %d samples across sources: %s", len(dataset), dict(source_counts))

    # Balance sources per configured `weight` (default 1.0 = equal) rather than
    # letting raw record counts dominate — e.g. mathv360k (~4000) would otherwise
    # drown out egoschema/vsr (~1000 each) despite all three having weight 1.0.
    sample_weights = [
        dataset_cfg.get(src, {}).get("weight", 1.0) / source_counts[src]
        for src in dataset.sources
    ]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

    dataloader = DataLoader(
        dataset,
        batch_size=tcfg["per_device_train_batch_size"],
        sampler=sampler,
        num_workers=tcfg["dataloader_num_workers"],
        pin_memory=True,
    )

    # 4. Build trainer components
    lora = LoRAAdapter(config_path=model_config)
    loss_engine = KDLossEngine.from_config(model_config)
    trainer = DistillationTrainer(student, lora, loss_engine, config_path=train_config)

    # 5. Train
    logger.info("Starting training...")
    trainer.train(dataloader, max_steps=max_steps)
    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--train-config", default="config/training_config.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    run_phase2(args.model_config, args.train_config, max_steps=args.max_steps)

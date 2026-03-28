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

import yaml
from torch.utils.data import DataLoader

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
) -> None:
    logger.info("=== Phase 2: Focused Student Training ===")

    with open(train_config) as f:
        tcfg = yaml.safe_load(f)["training"]

    # 1. Load cached teacher logits
    logger.info("Loading logit cache...")
    cache = LogitCache.from_config(train_config)
    num_records = cache.count()
    logger.info("Cache contains %d distillation records.", num_records)

    # 2. Build student model
    logger.info("Initialising student model...")
    student = StudentModel(config_path=model_config)

    # 3. Build dataset and dataloader
    dataset = DistillationDataset(
        cache=cache,
        processor=student.processor,
        max_seq_length=tcfg["max_seq_length"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=tcfg["per_device_train_batch_size"],
        shuffle=True,
        num_workers=tcfg["dataloader_num_workers"],
        pin_memory=True,
    )
    logger.info("Dataset size: %d samples", len(dataset))

    # 4. Build trainer components
    lora = LoRAAdapter(config_path=model_config)
    loss_engine = KDLossEngine.from_config(model_config)
    trainer = DistillationTrainer(student, lora, loss_engine, config_path=train_config)

    # 5. Train
    logger.info("Starting training...")
    trainer.train(dataloader)
    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--train-config", default="config/training_config.yaml")
    args = parser.parse_args()
    run_phase2(args.model_config, args.train_config)

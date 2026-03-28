from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from distillation.kd_loss_engine import KDLossEngine
from student.lora_adapter import LoRAAdapter
from student.student_model import StudentModel

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """
    Training loop for Phase 2: Focused Student Training.

    Combines:
        - Cross-entropy loss on ground-truth answers.
        - Reasoning-weighted KD loss from cached teacher logits.
        - LoRA-based parameter-efficient fine-tuning.
    """

    def __init__(
        self,
        student: StudentModel,
        lora: LoRAAdapter,
        loss_engine: KDLossEngine,
        config_path: str = "config/training_config.yaml",
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)["training"]

        self.student = student
        self.lora = lora
        self.loss_engine = loss_engine
        self.output_dir = Path(self.cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inject LoRA adapters
        self.peft_model = self.lora.injectAdapters(self.student.model)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.peft_model.parameters()),
            lr=float(self.cfg["learning_rate"]),
            weight_decay=float(self.cfg["weight_decay"]),
        )

    def train(self, dataloader: DataLoader) -> None:
        """
        Run the full training loop.

        Args:
            dataloader : yields batches with keys:
                         input_ids, labels, teacher_indices,
                         teacher_logprobs, reasoning_mask
        """
        num_epochs = self.cfg["num_epochs"]
        grad_accum = self.cfg["gradient_accumulation_steps"]
        total_steps = len(dataloader) * num_epochs // grad_accum

        warmup_steps = int(total_steps * self.cfg["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        device = next(self.peft_model.parameters()).device
        self.peft_model.train()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            self.optimizer.zero_grad()

            for step, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}

                logits = self.peft_model(
                    input_ids=batch["input_ids"],
                    attention_mask=(batch["input_ids"] != 0).long(),
                ).logits

                losses = self.loss_engine.compute(
                    student_logits=logits,
                    labels=batch["labels"],
                    teacher_indices=batch["teacher_indices"],
                    teacher_logprobs=batch["teacher_logprobs"],
                    reasoning_mask=batch["reasoning_mask"],
                )

                loss = losses["loss"] / grad_accum
                loss.backward()
                epoch_loss += losses["loss"].item()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.peft_model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.cfg["logging_steps"] == 0:
                        logger.info(
                            "Step %d | loss=%.4f  ce=%.4f  kd=%.4f  lr=%.2e",
                            global_step,
                            losses["loss"].item(),
                            losses["ce_loss"].item(),
                            losses["kd_loss"].item(),
                            scheduler.get_last_lr()[0],
                        )

                    if global_step % self.cfg["save_steps"] == 0:
                        ckpt_path = self.output_dir / f"checkpoint-{global_step}"
                        self.lora.save_adapter(self.peft_model, str(ckpt_path))
                        logger.info("Saved adapter checkpoint to %s", ckpt_path)

            avg = epoch_loss / len(dataloader)
            logger.info("Epoch %d/%d complete — avg loss: %.4f", epoch + 1, num_epochs, avg)

        # Final save
        final_path = self.output_dir / "final_adapter"
        self.lora.save_adapter(self.peft_model, str(final_path))
        logger.info("Training complete. Final adapter saved to %s", final_path)

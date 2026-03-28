from __future__ import annotations

from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel


class LoRAAdapter:
    """
    Injects LoRA low-rank matrices into the student model and manages
    adapter weight updates and persistence.

    Attributes (from class diagram):
        rank          : LoRA rank r.
        scalingFactor : lora_alpha / rank.

    Methods (from class diagram):
        injectAdapters()       : wrap the base model with PEFT LoRA config.
        updateLowRankWeights() : merge adapter deltas back into base weights.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            lora_cfg = yaml.safe_load(f)["lora"]

        self.rank: int = lora_cfg["rank"]
        self.lora_alpha: int = lora_cfg["lora_alpha"]
        self.scalingFactor: float = lora_cfg["scaling_factor"]
        self.lora_dropout: float = lora_cfg["lora_dropout"]
        self.target_modules: list[str] = lora_cfg["target_modules"]
        self.bias: str = lora_cfg["bias"]

    def injectAdapters(self, base_model: PreTrainedModel) -> PeftModel:
        """
        Wrap the base model with LoRA adapters.
        Only the adapter parameters will be trainable; base weights are frozen.

        Returns:
            peft_model : PeftModel with LoRA adapters injected.
        """
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model

    def updateLowRankWeights(
        self, peft_model: PeftModel, save_path: str
    ) -> PreTrainedModel:
        """
        Merge LoRA delta weights into the base model weights and save.
        The merged model can be deployed without the PEFT library.

        Args:
            peft_model : Trained PeftModel.
            save_path  : Directory to save the merged model.

        Returns:
            merged_model : Merged PreTrainedModel.
        """
        merged = peft_model.merge_and_unload()
        Path(save_path).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(save_path)
        return merged

    def save_adapter(self, peft_model: PeftModel, save_path: str) -> None:
        """Save only the LoRA adapter weights (lightweight checkpoint)."""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(save_path)

    def load_adapter(
        self, base_model: PreTrainedModel, adapter_path: str
    ) -> PeftModel:
        """Load a previously saved LoRA adapter onto a base model."""
        return PeftModel.from_pretrained(base_model, adapter_path)

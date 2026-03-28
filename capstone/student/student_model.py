from __future__ import annotations

import torch
import yaml
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class StudentModel:
    """
    Wraps Qwen2.5-VL-3B-Instruct (or 7B) as the deployable student model.

    Methods (matching class diagram):
        forward()  : returns logits for a batch — used during training.
        predict()  : runs greedy generation and returns decoded text — used at eval.

    LoRA adapters are injected externally by LoRAAdapter after construction.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["student"]

        self.model_id: str = cfg["model_id"]
        self.max_new_tokens: int = cfg["max_new_tokens"]

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=cfg.get("device_map", "auto"),
            torch_dtype=torch.bfloat16,
        )
        self.model.train()

    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, seq_len)
        attention_mask: torch.Tensor,  # (B, seq_len)
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run a forward pass and return per-token logits.

        Returns:
            logits : (B, seq_len, vocab_size)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.logits

    @torch.no_grad()
    def predict(
        self, question: str, image: Image.Image | None = None
    ) -> str:
        """
        Run inference and return the decoded answer string.
        Used during evaluation (model must be in eval() mode beforehand).
        """
        self.model.eval()
        inputs = self._build_inputs(question, image)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        self.model.train()
        return self.processor.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ #

    def _build_inputs(
        self, question: str, image: Image.Image | None
    ) -> dict[str, torch.Tensor]:
        if image is not None:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_prompt],
            images=[image] if image else None,
            return_tensors="pt",
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

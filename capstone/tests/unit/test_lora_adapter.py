import pytest
from unittest.mock import MagicMock, patch


class TestLoRAAdapter:
    """
    Unit tests for LoRAAdapter.

    These tests mock the PEFT library so no GPU or real model is required.
    """

    def _make_adapter(self):
        # Patch file open + yaml.safe_load so no real config file is needed
        import yaml
        from unittest.mock import mock_open, patch

        cfg = {
            "lora": {
                "rank": 16,
                "lora_alpha": 32,
                "scaling_factor": 2.0,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
        }
        with patch("builtins.open", mock_open(read_data="")):
            with patch("yaml.safe_load", return_value=cfg):
                from student.lora_adapter import LoRAAdapter
                return LoRAAdapter(config_path="dummy_config.yaml")

    def test_attributes_loaded_from_config(self):
        adapter = self._make_adapter()
        assert adapter.rank == 16
        assert adapter.lora_alpha == 32
        assert adapter.scalingFactor == 2.0

    def test_inject_adapters_calls_get_peft_model(self):
        adapter = self._make_adapter()
        fake_model = MagicMock()
        fake_peft_model = MagicMock()

        with patch("student.lora_adapter.get_peft_model", return_value=fake_peft_model) as mock_gpm:
            with patch("student.lora_adapter.LoraConfig") as mock_cfg:
                result = adapter.injectAdapters(fake_model)

        mock_gpm.assert_called_once()
        fake_peft_model.print_trainable_parameters.assert_called_once()
        assert result is fake_peft_model

    def test_update_low_rank_weights_merges_and_saves(self, tmp_path):
        adapter = self._make_adapter()
        fake_peft = MagicMock()
        fake_merged = MagicMock()
        fake_peft.merge_and_unload.return_value = fake_merged

        save_path = str(tmp_path / "merged")
        result = adapter.updateLowRankWeights(fake_peft, save_path)

        fake_peft.merge_and_unload.assert_called_once()
        fake_merged.save_pretrained.assert_called_once_with(save_path)
        assert result is fake_merged

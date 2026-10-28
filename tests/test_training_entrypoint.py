"""CPU-only orchestration checks; no model, accelerator, or training is run."""

import sys
from types import ModuleType

import pytest
import runtime
import torch
from omegaconf import OmegaConf
from utils import misc


@pytest.mark.parametrize("training_error", [False, True], ids=["success", "cleanup"])
def test_training_entrypoint_prepares_config_and_seed_before_model(
    tmp_path, monkeypatch, training_error
):
    output = tmp_path / "fresh" / "fr3"
    cfg = OmegaConf.create(
        {
            "output_dir": str(output),
            "mixed_precision": "bf16",
            "gradient_accumulation_steps": 2,
            "seed": 123,
            "model": {"_target_": "unused", "initial_seed": "${seed}"},
            "data": {"train": "unused"},
            "vae_micro_batch_size": None,
            "inference_cross_kv_reuse": True,
        }
    )
    events = []
    model, train_dataset, val_dataset = object(), object(), object()

    class FakeAccelerator:
        is_main_process = True
        device = "cpu"

        def __init__(self, **kwargs):
            assert kwargs == {
                "gradient_accumulation_steps": 2,
                "mixed_precision": "bf16",
                "step_scheduler_with_optimizer": False,
            }
            events.append("accelerator")

        def end_training(self):
            events.append("end_training")

    def seed(value):
        assert value == 123
        events.append("seed")

    def instantiate_model(model_cfg, *, model_dtype, device):
        assert events == ["accelerator", "seed"]
        assert model_cfg is cfg.model
        assert model_dtype == torch.bfloat16 and device == "cpu"
        assert misc.get_work_dir() == str(output)
        assert output.is_dir()
        saved = OmegaConf.load(output / "config.yaml")
        assert OmegaConf.to_container(saved) == OmegaConf.to_container(
            cfg, resolve=True
        )
        assert saved.model.initial_seed == 123
        events.append("model")
        return model

    def configure_model(actual, **kwargs):
        assert actual is model
        assert kwargs == {
            "vae_micro_batch_size": None,
            "inference_cross_kv_reuse": True,
        }
        events.append("configure")
        return model

    def build_datasets(data_cfg):
        assert data_cfg is cfg.data
        events.append("datasets")
        return train_dataset, val_dataset

    class FakeTrainer:
        def __init__(self, **kwargs):
            assert kwargs["cfg"] is cfg
            assert isinstance(kwargs["accelerator"], FakeAccelerator)
            assert kwargs["model"] is model
            assert kwargs["train_dataset"] is train_dataset
            assert kwargs["val_dataset"] is val_dataset
            events.append("trainer")

        def train(self):
            events.append("train")
            if training_error:
                raise RuntimeError("synthetic trainer failure")

    accelerate = ModuleType("accelerate")
    accelerate.Accelerator = FakeAccelerator
    accelerate_utils = ModuleType("accelerate.utils")
    accelerate_utils.set_seed = seed
    trainer = ModuleType("trainer")
    trainer.EasyWAMTrainer = FakeTrainer
    inference = ModuleType("model.helpers.inference")
    inference.configure_model_execution = configure_model
    for name, module in (
        ("accelerate", accelerate),
        ("accelerate.utils", accelerate_utils),
        ("trainer", trainer),
        ("model.helpers.inference", inference),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(misc, "_WORK_DIR", None)
    monkeypatch.setattr(runtime, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "instantiate", instantiate_model)
    monkeypatch.setattr(runtime, "build_datasets", build_datasets)

    assert not output.exists()
    if training_error:
        with pytest.raises(RuntimeError, match="synthetic trainer failure"):
            runtime.run_training(cfg)
    else:
        runtime.run_training(cfg)
    assert events == [
        "accelerator",
        "seed",
        "model",
        "configure",
        "datasets",
        "trainer",
        "train",
        "end_training",
    ]

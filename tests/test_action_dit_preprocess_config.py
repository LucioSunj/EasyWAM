"""Check real Hydra composition and tiny, fake ActionDiT payload publication."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import preprocess_action_dit_backbone as preprocess

ROOT = Path(__file__).resolve().parents[1]
TASK = "real_world_fr3_fastwam_idm_wan22"


def test_legacy_model_config_keeps_backbone_initialization_defaults():
    backbone, action = preprocess._load_model_config(
        ROOT / "configs/model/easywam_mot_wan22.yaml", "wan22"
    )

    assert backbone["name"] == "wan22"
    assert backbone["dit_path"] is None
    assert backbone["text_encoder_path"] is None
    assert backbone["vae_path"] is None
    assert action["action_dim"] == 7
    assert action["hidden_dim"] == 1024 and action["num_layers"] == 30


def test_task_composition_resolves_fr3_dimensions_and_final_asset_overrides(tmp_path):
    robot = str(tmp_path / "robot/checkpoint.safetensors")
    text = str(tmp_path / "base/umt5.pth")
    vae = str(tmp_path / "base/vae.pth")
    overrides = [
        f"task={TASK}",
        f"data.train.dataset_dirs=[{tmp_path / 'data'}]",
        "model.backbone.dit_path=earlier.safetensors",
        f"model.backbone.dit_path={robot}",
        f"model.backbone.text_encoder_path={text}",
        f"model.backbone.vae_path={vae}",
        f"output_dir={tmp_path / 'run'}",
        "batch_size=2",
    ]

    backbone, action = preprocess._load_model_config(None, None, overrides, task=TASK)

    assert action["action_dim"] == 8
    assert backbone["dit_path"] == robot
    assert backbone["text_encoder_path"] == text
    assert backbone["vae_path"] == vae
    assert not (tmp_path / "run").exists()


def test_task_cli_uses_final_config_and_writes_tiny_interpolated_payload(
    tmp_path, monkeypatch
):
    output = tmp_path / "action.pt"
    robot = str(tmp_path / "robot/checkpoint.safetensors")
    key = "blocks.0.self_attn.q.weight"
    seen = {}

    class FakeVideo:
        num_heads = 24
        attn_head_dim = 128
        blocks = [None] * 30

        def state_dict(self):
            return {key: torch.ones(4, 4)}

    class FakeAction:
        ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.",)

        def __init__(self, **kwargs):
            seen["action_config"] = kwargs
            self.num_heads = kwargs["num_heads"]
            self.attn_head_dim = kwargs["attn_head_dim"]
            self.num_layers = kwargs["num_layers"]

        def to(self, **kwargs):
            assert kwargs == {"device": "cpu", "dtype": torch.float32}
            return self

        def state_dict(self):
            return {key: torch.zeros(2, 2), "action_encoder.weight": torch.zeros(2, 8)}

        @staticmethod
        def backbone_key_set(keys):
            return {name for name in keys if not name.startswith("action_encoder.")}

    def fake_loader(cfg, **kwargs):
        seen["backbone_config"] = cfg
        assert kwargs == {"device": "cpu", "torch_dtype": torch.float32}
        return SimpleNamespace(dit=FakeVideo())

    monkeypatch.setattr(preprocess, "load_easywam_backbone", fake_loader)
    monkeypatch.setattr(preprocess, "ActionDiT", FakeAction)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_action_dit_backbone.py",
            "--task",
            TASK,
            "--override",
            "model.backbone.dit_path=earlier.safetensors",
            "--override",
            f"model.backbone.dit_path={robot}",
            "--output",
            str(output),
        ],
    )

    preprocess.main()

    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert seen["action_config"]["action_dim"] == 8
    assert seen["backbone_config"]["dit_path"] == robot
    assert payload["policy"]["source_backbone"] == "wan22"
    assert payload["policy"]["alpha_scaling"] is True
    assert set(payload["backbone_state_dict"]) == {key}
    torch.testing.assert_close(
        payload["backbone_state_dict"][key], torch.full((2, 2), math.sqrt(2))
    )
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("existing", [False, True])
def test_failed_save_does_not_publish_partial_payload(tmp_path, monkeypatch, existing):
    output = tmp_path / "action.pt"
    if existing:
        output.write_bytes(b"previous completed payload")

    def interrupted_save(payload, path):
        temporary = Path(path)
        assert temporary.parent == output.parent and temporary != output
        temporary.write_bytes(b"incomplete")
        raise RuntimeError("synthetic interrupted write")

    monkeypatch.setattr(preprocess.torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="synthetic interrupted write"):
        preprocess._save_payload_atomic({}, output)

    if existing:
        assert output.read_bytes() == b"previous completed payload"
        assert list(tmp_path.iterdir()) == [output]
    else:
        assert not output.exists()
        assert list(tmp_path.iterdir()) == []

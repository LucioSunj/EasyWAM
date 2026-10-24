"""Route local Wan assets without constructing models or reading weights."""

from types import SimpleNamespace

import pytest
import torch

from model.backbone.loader import load_easywam_backbone
from model.backbone.wan22 import loader


def create_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def test_default_paths_keep_sharded_dit_and_colocated_assets(tmp_path):
    first = create_file(tmp_path / "diffusion_pytorch_model-00001.safetensors")
    second = create_file(tmp_path / "diffusion_pytorch_model-00002.safetensors")
    text = create_file(tmp_path / "models_t5_umt5-xxl-enc-bf16.pth")
    vae = create_file(tmp_path / "Wan2.2_VAE.pth")
    tokenizer = tmp_path / "google/umt5-xxl"
    tokenizer.mkdir(parents=True)

    configs = loader._resolve_configs(str(tmp_path), str(tokenizer))

    assert [config.resolve() for config in configs] == [
        [first, second],
        text,
        vae,
        str(tokenizer),
    ]


def test_text_override_does_not_resolve_unused_video_assets(tmp_path):
    text = create_file(tmp_path / "text/encoder.pth")
    dit, encoder, vae, _ = loader._resolve_configs(
        str(tmp_path / "absent_robot_bundle"),
        str(tmp_path / "unused_tokenizer"),
        text_encoder_path=text,
    )

    assert encoder.resolve() == text
    assert dit.path is None and vae.path is None


def test_backbone_loader_routes_separate_robot_text_and_vae_files(
    tmp_path, monkeypatch
):
    dit_path = create_file(tmp_path / "robot/checkpoint.safetensors")
    text_path = create_file(tmp_path / "text/umt5.pth")
    vae_path = create_file(tmp_path / "vae/wan.pth")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    calls = []
    models = {}

    def load_model(path, model_name, **kwargs):
        calls.append((path, model_name, kwargs))
        result = object()
        models[model_name] = result
        return result

    def tokenizer(**kwargs):
        assert kwargs == {
            "name": str(tokenizer_path),
            "seq_len": 128,
            "clean": "whitespace",
        }
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(loader, "_load_registered_model", load_model)
    monkeypatch.setattr(loader, "_validate_dit_config", lambda config: config)
    monkeypatch.setattr(loader, "HuggingfaceTokenizer", tokenizer)
    cfg = {
        "name": "wan22",
        "model_id": str(tmp_path / "bundle_without_default_files"),
        "tokenizer_model_id": str(tokenizer_path),
        "dit_path": dit_path,
        "text_encoder_path": text_path,
        "vae_path": vae_path,
        "dit_config": {"text_dim": 4096},
        "text_dim": 4096,
        "load_text_encoder": True,
    }

    components = load_easywam_backbone(cfg, device="cpu", torch_dtype=torch.bfloat16)

    assert [(path, name) for path, name, _ in calls] == [
        (dit_path, "wan_video_dit"),
        (text_path, "wan_video_text_encoder"),
        (vae_path, "wan_video_vae"),
    ]
    assert all(call[2]["device"] == "cpu" for call in calls)
    assert components.dit is models["wan_video_dit"]
    assert components.text_encoder is models["wan_video_text_encoder"]
    assert components.vae is models["wan_video_vae"]
    assert components.dit_path == dit_path
    assert components.text_encoder_path == text_path
    assert components.vae_path == vae_path


def test_missing_explicit_vae_fails_before_loading_any_model(tmp_path, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Missing VAE must fail before model construction.")

    monkeypatch.setattr(loader, "_validate_dit_config", lambda config: config)
    monkeypatch.setattr(loader, "_load_registered_model", unexpected_load)
    with pytest.raises(FileNotFoundError, match="missing_vae.pth"):
        loader.load_wan22_ti2v_5b_components(
            device="cpu",
            model_id=str(tmp_path),
            dit_config={},
            load_text_encoder=False,
            vae_path=str(tmp_path / "missing_vae.pth"),
        )

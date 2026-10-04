"""CPU tests for the FastWAM model family on EasyWAM infrastructure."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastwam_fixtures import ACTION_CONFIG, TinyVAE, make_model, make_sample
from hydra import compose, initialize_config_dir
from torch import nn

from model.component.action_dit import ActionDiT, StateEncoder
from model.component.attention import StructuredAttentionMask
from model.easywam_mot import EasyWAMMoT
from model.easywam_mot_idm import EasyWAMMoTIDM
from model.easywam_mot_joint import EasyWAMMoTJoint
from model.fastwam import FastWAM, FastWAMActionDiT, FastWAMIDM, FastWAMJoint
from model.helpers.inference import configure_model_execution
from trainer import EasyWAMTrainer, _forward_training_model

MODEL_CLASSES = (FastWAM, FastWAMJoint, FastWAMIDM)
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_training_updates_linear_state_and_action_with_easywam_trainer(
    model_cls, checkpointing
):
    model = make_model(model_cls, checkpointing=checkpointing)
    EasyWAMTrainer._apply_dit_only_train_mode(model)
    assert isinstance(model.state_encoder, nn.Linear)
    assert isinstance(model.action_expert.action_encoder, nn.Linear)
    assert isinstance(model.action_expert.head, nn.Linear)
    assert not any(p.requires_grad for p in model.vae.parameters())
    assert not any(p.requires_grad for p in model.text_encoder.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    original = model.state_encoder.weight.detach().clone()
    loss, metrics = _forward_training_model(model, make_sample())
    assert torch.isfinite(loss)
    assert {"loss_video", "loss_action"} <= metrics.keys()
    loss.backward()
    for parameter in (model.state_encoder.weight, model.action_expert.head.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(original, model.state_encoder.weight)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_text_padding_is_masked_in_training_and_online_encoding(model_cls):
    model = make_model(model_cls)
    context, mask = model.encode_prompt(["short", "long"])
    assert mask.tolist() == [[True, True, False, False], [True, True, True, False]]
    sample = make_sample()
    sample.update(context=context, context_mask=mask)
    changed = dict(sample, context=context.clone())
    changed["context"][~mask] = 1000.0
    torch.manual_seed(42)
    loss = model(sample)[0]
    torch.manual_seed(42)
    changed_loss = model(changed)[0]
    torch.testing.assert_close(loss, changed_loss, atol=1e-6, rtol=1e-6)
    torch.manual_seed(42)
    visible_padding_loss = model(dict(changed, context_mask=torch.ones_like(mask)))[0]
    assert not torch.isclose(loss, visible_padding_loss, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_legacy_checkpoint_and_native_roundtrip(tmp_path, model_cls):
    source = make_model(model_cls)
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {
            "mot": source.mot.state_dict(),
            "proprio_encoder": source.state_encoder.state_dict(),
            "step": 17,
        },
        legacy,
    )
    target = make_model(model_cls)
    with torch.no_grad():
        target.action_expert.head.weight.zero_()
        target.state_encoder.weight.zero_()
    target.load_checkpoint(legacy)
    for name, value in source.mot.state_dict().items():
        torch.testing.assert_close(target.mot.state_dict()[name], value, atol=0, rtol=0)
    torch.testing.assert_close(
        source.state_encoder.weight, target.state_encoder.weight, atol=0, rtol=0
    )
    native = tmp_path / "native.pt"
    target.save_checkpoint(native, step=18)
    payload = target.load_checkpoint(native)
    assert payload["step"] == 18
    assert payload["text_padding"] == "masked"
    assert payload["model_variant"] == model_cls.model_variant
    assert "proprio_encoder" not in payload
    missing = dict(payload, mot=dict(payload["mot"]))
    missing["mot"].pop("mixtures.action.head.weight")
    torch.save(missing, tmp_path / "missing.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        target.load_checkpoint(tmp_path / "missing.pt")


def test_checkpoint_rejects_wrong_variant(tmp_path):
    path = tmp_path / "joint.pt"
    make_model(FastWAMJoint).save_checkpoint(path)
    with pytest.raises(ValueError, match="model_variant"):
        make_model(FastWAM).load_checkpoint(path)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_batched_inference_matches_independent_requests_and_masks_padding(model_cls):
    model = make_model(model_cls).eval()
    configure_model_execution(
        model, vae_micro_batch_size=None, inference_cross_kv_reuse=True
    )
    sample = make_sample()
    kwargs = {
        "prompt": None,
        "input_image": sample["video"][:, :, 0],
        "proprio": sample["proprio"][:, 0],
        "context": sample["context"],
        "context_mask": sample["context_mask"],
        "action_horizon": 4,
        "num_inference_steps": 2,
        "seed": [51, 89],
        "rand_device": "cpu",
    }
    if model_cls is not FastWAM:
        kwargs["num_video_frames"] = 5
    batched = model.infer_action_batch(**kwargs)["action"]
    for i, seed in enumerate(kwargs["seed"]):
        single = {
            key: value[i : i + 1] if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        single["seed"] = seed
        expected = model.infer_action_batch(**single)["action"]
        torch.testing.assert_close(batched[i : i + 1], expected, atol=2e-5, rtol=2e-5)
    changed_context = kwargs["context"].clone()
    changed_context[~kwargs["context_mask"]] = -1000.0
    actual = model.infer_action_batch(**dict(kwargs, context=changed_context))["action"]
    torch.testing.assert_close(batched, actual, atol=2e-5, rtol=2e-5)
    model.inference_cross_kv_reuse = False
    uncached = model.infer_action_batch(**kwargs)["action"]
    torch.testing.assert_close(batched, uncached, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_native_factory_builds_fastwam_components(monkeypatch, model_cls):
    import runtime
    from model.backbone import loader

    reference = make_model(model_cls)
    components = SimpleNamespace(
        dit=reference.video_expert,
        vae=TinyVAE(),
        text_encoder=None,
        tokenizer=None,
        text_dim=12,
        block_protocol="main",
        dit_path="synthetic",
        vae_path="synthetic",
        text_encoder_path=None,
        tokenizer_path=None,
    )
    monkeypatch.setattr(loader, "load_easywam_backbone", lambda *a, **kw: components)
    model = getattr(runtime, "create_" + model_cls.model_variant)(
        backbone={"name": "wan22"},
        action_dit_config=ACTION_CONFIG,
        state_dim=4,
        device="cpu",
        model_dtype=torch.float32,
    )
    assert type(model) is model_cls
    assert isinstance(model.action_expert, FastWAMActionDiT)
    assert isinstance(model.state_encoder, nn.Linear)
    mask = model._build_mot_attention_mask(6, 4, 2, torch.device("cpu"))
    assert isinstance(mask, StructuredAttentionMask)
    assert FastWAMActionDiT.backbone_key_set(
        model.action_expert.state_dict()
    ) == ActionDiT.backbone_key_set(ActionDiT(**ACTION_CONFIG).state_dict())


@pytest.mark.parametrize("dataset", ["libero", "robotwin"])
@pytest.mark.parametrize("variant", ["fastwam", "fastwam_joint", "fastwam_idm"])
def test_training_and_evaluation_recipes_compose(dataset, variant):
    task = f"{dataset}_{variant}_wan22"
    for config_name in ("train", f"benchmark/sim_{dataset}"):
        with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.3"):
            cfg = compose(config_name=config_name, overrides=[f"task={task}"])
        assert cfg.model._target_ == f"runtime.create_{variant}"
        assert cfg.model.state_position == "context"
        assert cfg.model.backbone.text_encoder_id == "wan22ti2v5b"
        assert cfg.model.action_dit_config.num_layers == 30
        assert (
            cfg.model.action_dit_config.action_dim
            == cfg.data.train.processor.action_output_dim
        )


@pytest.mark.parametrize("model_cls", [EasyWAMMoT, EasyWAMMoTJoint, EasyWAMMoTIDM])
def test_existing_easywam_models_keep_mlp_projections(model_cls):
    model = make_model(model_cls)
    assert isinstance(model.state_encoder, StateEncoder)
    assert not isinstance(model.action_expert.action_encoder, nn.Linear)
    assert torch.isfinite(model(make_sample())[0])


@pytest.mark.parametrize(
    "model_name,backbone_name,layers,text_dim",
    [
        ("fastwam_wan22", "wan22", 30, 4096),
        ("fastwam_joint_wan22", "wan22", 30, 4096),
        ("fastwam_idm_wan22", "wan22", 30, 4096),
        ("easywam_mot_wan22", "wan22", 30, 4096),
        ("easywam_mot_wan22_lora", "wan22", 30, 4096),
        ("easywam_mot_cosmos25", "cosmos25", 28, 1024),
        ("easywam_mot_cosmos25_lora", "cosmos25", 28, 1024),
    ],
)
def test_action_backbone_preparation_resolves_inherited_model_config(
    model_name, backbone_name, layers, text_dim
):
    from scripts.preprocess_action_dit_backbone import _load_model_config

    backbone, action = _load_model_config(
        CONFIG_ROOT / "model" / f"{model_name}.yaml", backbone_name
    )
    assert action["action_dim"] == 7
    assert action["num_layers"] == backbone["num_layers"] == layers
    assert action["text_dim"] == backbone["text_dim"] == text_dim
    assert action["use_gradient_checkpointing"] is False
    if backbone_name == "wan22":
        assert backbone["dit_config"]["use_gradient_checkpointing"] is False


def test_native_training_entrypoint_saves_and_resumes_cpu_state(tmp_path, monkeypatch):
    import json

    import runtime

    sample = make_sample()
    dataset = [{key: value[i] for key, value in sample.items()} for i in range(2)]
    monkeypatch.setattr(runtime, "build_datasets", lambda cfg: (dataset, None))
    monkeypatch.setattr(runtime, "create_fastwam", lambda **kw: make_model(FastWAM))
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=libero_fastwam_wan22",
                f"output_dir={tmp_path}",
                "batch_size=2",
                "num_workers=0",
                "mixed_precision=no",
                "max_steps=1",
                "save_every=1",
                "log_every=1",
                "eval_every=0",
            ],
        )
    runtime.run_training(cfg)
    state_dir = tmp_path / "checkpoints/state/step_000001"
    trainer_state = json.loads((state_dir / "trainer_state.json").read_text())
    assert trainer_state["global_step"] == 1
    first = torch.load(
        tmp_path / "checkpoints/weights/step_000001.pt", weights_only=True
    )
    cfg.resume = str(state_dir)
    cfg.max_steps = 2
    runtime.run_training(cfg)
    second = torch.load(
        tmp_path / "checkpoints/weights/step_000002.pt", weights_only=True
    )
    assert second["step"] == 2 and second["text_padding"] == "masked"
    assert not torch.equal(
        first["mot"]["mixtures.action.head.weight"],
        second["mot"]["mixtures.action.head.weight"],
    )


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_dynamic_batcher_preserves_predictions_with_trimmed_prompt_padding(model_cls):
    from concurrent.futures import ThreadPoolExecutor

    from experiments.batched_inference import DynamicInferenceBatcher

    model = make_model(model_cls).eval()
    sample = make_sample()
    options = {"action_horizon": 4, "num_inference_steps": 2, "rand_device": "cpu"}
    if model_cls is not FastWAM:
        options["num_video_frames"] = 5
    expected = model.infer_action_batch(
        prompt=["short", "long"],
        input_image=sample["video"][:, :, 0],
        proprio=sample["proprio"][:, 0],
        seed=[51, 89],
        **options,
    )["action"]
    with DynamicInferenceBatcher(
        model, max_batch_size=2, wait_ms=30, prompt_cache_size=2
    ) as batcher:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    batcher.submit,
                    "action",
                    prompt=prompt,
                    input_image=sample["video"][i, :, 0],
                    proprio=sample["proprio"][i, 0],
                    seed=seed,
                    **options,
                )
                for i, (prompt, seed) in enumerate(zip(["short", "long"], [51, 89]))
            ]
            actual = torch.stack([future.result()["action"] for future in futures])
        assert batcher.stats()["requests"] == 2
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

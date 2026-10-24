import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from fastwam_fixtures import ACTION_CONFIG, TinyVAE, make_model, make_sample
from model.component.mot import MoT
from model.fastwam import FastWAMIDM
from trainer import EasyWAMTrainer


@pytest.mark.parametrize("layers", [None, [], [0], [1], [0, 1]])
def test_selected_checkpoint_calls_and_gradient_parity(monkeypatch, layers):
    torch.set_num_threads(1)
    reference = make_model(FastWAMIDM, checkpointing=False)
    selected = make_model(FastWAMIDM, checkpointing=True)
    selected.mot = MoT({"video": selected.video_expert, "action": selected.action_expert},
                       gradient_checkpointing_layers=layers)
    selected.load_state_dict(reference.state_dict())
    calls = []
    original = torch.utils.checkpoint.checkpoint

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counted)
    losses = []
    for model in (reference, selected):
        EasyWAMTrainer._apply_dit_only_train_mode(model)
        torch.manual_seed(123)
        loss, _ = model(copy.deepcopy(make_sample()))
        loss.backward()
        losses.append(loss.detach())
    assert len(calls) == (2 if layers is None else len(layers))
    assert all(call["use_reentrant"] is False for call in calls)
    torch.testing.assert_close(losses[0], losses[1])
    actual = dict(selected.named_parameters())
    for name, parameter in reference.named_parameters():
        other = actual[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(other.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad, msg=lambda message: name + ": " + message)


@pytest.mark.parametrize("layers", [[-1], [2], [0, 0], [True], [0.5], ["0"]])
def test_invalid_layer_selection_is_rejected(layers):
    model = make_model(FastWAMIDM)
    with pytest.raises(ValueError, match="layer indices"):
        MoT({"video": model.video_expert, "action": model.action_expert},
            gradient_checkpointing_layers=layers)


def test_resolved_backbone_config_reaches_mot(monkeypatch):
    import runtime
    from model.backbone import loader

    reference = make_model(FastWAMIDM, checkpointing=True)
    components = SimpleNamespace(dit=reference.video_expert, vae=TinyVAE(), text_encoder=None,
        tokenizer=None, text_dim=12, block_protocol="main", dit_path="synthetic",
        vae_path="synthetic", text_encoder_path=None, tokenizer_path=None)
    monkeypatch.setattr(loader, "load_easywam_backbone", lambda *args, **kwargs: components)
    model = runtime.create_fastwam_idm(
        backbone=OmegaConf.create({"name": "wan22", "mot_gradient_checkpointing_layers": [0]}),
        action_dit_config=dict(ACTION_CONFIG, use_gradient_checkpointing=True), state_dim=4,
        device="cpu", model_dtype=torch.float32)
    assert model.mot.gradient_checkpointing_layers == frozenset({0})

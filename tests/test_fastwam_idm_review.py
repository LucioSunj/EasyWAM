"""Information-flow acceptance tests for the real-world IDM training path."""

import pytest
import torch

from fastwam_fixtures import make_model, make_sample
from model.fastwam import FastWAMIDM
from trainer import EasyWAMTrainer


def test_teacher_forcing_mask_keeps_video_branches_independent():
    model = make_model(FastWAMIDM)
    mask = model._build_teacher_forcing_attention_mask(6, 6, 4, 2, 2, torch.device('cpu')).dense
    assert mask.shape == (16, 16)
    assert not mask[:6, 6:].any()
    assert not mask[6:12, :6].any() and not mask[6:12, 12:].any()
    assert not mask[12:, :6].any()
    assert mask[12:, 6:].all()
    assert mask[:2, :2].all() and not mask[:2, 2:].any()
    assert mask[6:8, 6:8].all() and not mask[6:8, 8:].any()


@pytest.mark.parametrize('objective', ['action', 'video'])
def test_idm_loss_gradient_routes_match_the_attention_graph(objective):
    torch.set_num_threads(1)
    model = make_model(FastWAMIDM)
    EasyWAMTrainer._apply_dit_only_train_mode(model)
    model.loss_lambda_action = float(objective == 'action')
    model.loss_lambda_video = float(objective == 'video')
    captured = {}

    def capture_inputs(module, args, kwargs):
        for key, value in kwargs['embeds_all'].items():
            value.retain_grad()
            captured[key] = value

    handle = model.mot.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    loss, _ = model(make_sample())
    loss.backward()
    handle.remove()
    video_grad = captured['video'].grad
    split = video_grad.shape[1] // 2
    assert torch.isfinite(video_grad).all()
    if objective == 'action':
        assert video_grad[:, :split].count_nonzero() == 0
        assert video_grad[:, split:].abs().sum() > 0
        assert captured['action'].grad.abs().sum() > 0
        assert model.state_encoder.weight.grad.abs().sum() > 0
    else:
        assert video_grad[:, :split].abs().sum() > 0
        assert video_grad[:, split:].count_nonzero() == 0
        assert captured['action'].grad.count_nonzero() == 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.video_dit.parameters())
    assert all(p.grad is None for p in model.vae.parameters())
    assert all(p.grad is None for p in model.text_encoder.parameters())


@pytest.mark.parametrize('objective', ['action', 'video'])
def test_idm_fully_padded_targets_have_zero_selected_loss(objective):
    model = make_model(FastWAMIDM)
    sample = make_sample()
    sample['action_is_pad'].fill_(True)
    sample['image_is_pad'].fill_(True)
    model.loss_lambda_action = float(objective == 'action')
    model.loss_lambda_video = float(objective == 'video')
    loss, metrics = model(sample)
    assert torch.isfinite(loss)
    assert loss.item() == 0
    assert metrics[f'loss_{objective}'].item() == 0


@pytest.mark.parametrize('padded_field', ['action', 'video'])
def test_padded_values_cannot_change_valid_idm_losses(padded_field):
    model = make_model(FastWAMIDM)
    sample = make_sample()
    sample['action_is_pad'] = torch.tensor([[False, True, True, True], [False, False, True, True]])
    sample['image_is_pad'] = torch.tensor([[False, True, True, True, True], [False, False, False, True, True]])
    changed = dict(sample)
    if padded_field == 'action':
        changed['action'] = torch.where(sample['action_is_pad'][..., None], sample['action']+.25, sample['action'])
    else:
        changed['video'] = torch.where(sample['image_is_pad'][:, None, :, None, None], sample['video']+.25, sample['video'])
    torch.manual_seed(51)
    loss, metrics = model(sample)
    torch.manual_seed(51)
    changed_loss, changed_metrics = model(changed)
    torch.testing.assert_close(changed_loss, loss, atol=1e-6, rtol=1e-6)
    for key in ('loss_action', 'loss_video'):
        torch.testing.assert_close(changed_metrics[key], metrics[key], atol=1e-6, rtol=1e-6)


def test_padded_video_canonicalization_handles_leading_internal_and_all_padding():
    video = torch.arange(10, dtype=torch.float32).reshape(2, 1, 5, 1, 1)
    padding = torch.tensor([[True, False, True, False, True], [True]*5])
    sample = {'video': video, 'image_is_pad': padding}
    canonical = FastWAMIDM._canonicalize_padded_video(sample)['video']
    torch.testing.assert_close(canonical.flatten(), torch.tensor([0., 1., 1., 3., 3., 0., 0., 0., 0., 0.]))
    torch.testing.assert_close(video.flatten(), torch.arange(10, dtype=torch.float32))

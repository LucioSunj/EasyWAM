"""Regression coverage for evaluation padding field flow and reductions."""

import pytest
import torch
from PIL import Image

from fastwam_fixtures import make_model, make_sample
from model.fastwam import FastWAMIDM
from trainer import EasyWAMTrainer, _canonicalize_eval_video
from utils.video_metrics import masked_action_errors, video_psnr, video_ssim


def _unbatched_eval_sample():
    sample = make_sample()
    return {
        "video": sample["video"][0],
        "action": sample["action"][0],
        "proprio": sample["proprio"][0],
        "prompt": "short",
        "context": sample["context"][0],
        "context_mask": sample["context_mask"][0],
        "image_is_pad": torch.tensor([False, False, True, True, True]),
        "action_is_pad": torch.tensor([False, False, True, True]),
        "proprio_is_pad": torch.tensor([False, False, True, True, True]),
        "action_dim_is_pad": torch.tensor([False, True, False]),
        "proprio_dim_is_pad": torch.tensor([False, False, True, False]),
    }


def test_eval_adapter_batches_temporal_and_dimension_masks():
    sample = EasyWAMTrainer._to_batched_eval_sample(_unbatched_eval_sample())

    assert sample["video"].shape == (1, 3, 5, 16, 32)
    assert sample["action"].shape == (1, 4, 3)
    assert sample["proprio"].shape == (1, 5, 4)
    assert sample["image_is_pad"].shape == (1, 5)
    assert sample["action_is_pad"].shape == (1, 4)
    assert sample["proprio_is_pad"].shape == (1, 5)
    assert sample["action_dim_is_pad"].shape == (1, 3)
    assert sample["proprio_dim_is_pad"].shape == (1, 4)

    batched = dict(_unbatched_eval_sample())
    batched["video"] = batched["video"].unsqueeze(0)
    batched["action"] = batched["action"].unsqueeze(0)
    batched["proprio"] = batched["proprio"].unsqueeze(0)
    batched["context"] = batched["context"].unsqueeze(0)
    batched["context_mask"] = batched["context_mask"].unsqueeze(0)
    batched["prompt"] = ["short"]
    batched["image_is_pad"] = batched["image_is_pad"].unsqueeze(0)
    batched["action_is_pad"] = batched["action_is_pad"].unsqueeze(0)
    batched["proprio_is_pad"] = batched["proprio_is_pad"].unsqueeze(0)
    batched["action_dim_is_pad"] = batched["action_dim_is_pad"].unsqueeze(0)
    batched["proprio_dim_is_pad"] = batched["proprio_dim_is_pad"].unsqueeze(0)
    actual = EasyWAMTrainer._to_batched_eval_sample(batched)
    for key in (
        "image_is_pad",
        "action_is_pad",
        "proprio_is_pad",
        "action_dim_is_pad",
        "proprio_dim_is_pad",
    ):
        torch.testing.assert_close(actual[key], sample[key])


def test_eval_adapter_keeps_missing_masks_backward_compatible():
    sample = _unbatched_eval_sample()
    for key in (
        "image_is_pad",
        "action_is_pad",
        "proprio_is_pad",
        "action_dim_is_pad",
        "proprio_dim_is_pad",
    ):
        sample.pop(key)
    actual = EasyWAMTrainer._to_batched_eval_sample(sample)
    assert all(actual[key] is None for key in (
        "image_is_pad",
        "action_is_pad",
        "proprio_is_pad",
        "action_dim_is_pad",
        "proprio_dim_is_pad",
    ))


def test_eval_adapter_rejects_unbatched_masks_for_batched_values():
    sample = _unbatched_eval_sample()
    sample["video"] = sample["video"].unsqueeze(0).repeat(2, 1, 1, 1, 1)
    sample["action"] = sample["action"].unsqueeze(0).repeat(2, 1, 1)
    sample["proprio"] = sample["proprio"].unsqueeze(0).repeat(2, 1, 1)
    sample["context"] = sample["context"].unsqueeze(0).repeat(2, 1, 1)
    sample["context_mask"] = sample["context_mask"].unsqueeze(0).repeat(2, 1)
    sample["prompt"] = ["short", "short"]
    with pytest.raises(ValueError, match="batch size 2"):
        EasyWAMTrainer._to_batched_eval_sample(sample)


@pytest.mark.parametrize(
    "key,value",
    [
        ("image_is_pad", torch.zeros(4, dtype=torch.bool)),
        ("action_is_pad", torch.zeros(3, dtype=torch.bool)),
        ("proprio_is_pad", torch.zeros(4, dtype=torch.bool)),
        ("action_dim_is_pad", torch.zeros(2, dtype=torch.bool)),
        ("proprio_dim_is_pad", torch.zeros(3, dtype=torch.bool)),
    ],
)
def test_eval_adapter_rejects_invalid_mask_shapes(key, value):
    sample = _unbatched_eval_sample()
    sample[key] = value
    with pytest.raises(ValueError, match=key):
        EasyWAMTrainer._to_batched_eval_sample(sample)


def test_video_metrics_ignore_padded_frames_and_skip_all_padding():
    target = torch.zeros(3, 4, 16, 16)
    pred = target.clone()
    pred[:, 1] = 1.0
    pred[:, 3] = 1.0
    valid = torch.tensor([False, True, False, True])

    psnr, count = video_psnr(
        pred,
        target,
        valid_frames=valid,
        exclude_first_frame=True,
        return_count=True,
    )
    ssim, ssim_count = video_ssim(
        pred,
        target,
        valid_frames=valid,
        exclude_first_frame=True,
        return_count=True,
    )
    changed = pred.clone()
    changed[:, 0] = 100.0
    changed[:, 2] = -100.0
    changed_psnr, _ = video_psnr(
        changed,
        target,
        valid_frames=valid,
        exclude_first_frame=True,
        return_count=True,
    )
    changed_ssim, _ = video_ssim(
        changed,
        target,
        valid_frames=valid,
        exclude_first_frame=True,
        return_count=True,
    )
    assert count == ssim_count == 2
    assert psnr == changed_psnr
    assert ssim == changed_ssim

    skipped, skipped_count = video_psnr(
        pred,
        target,
        valid_frames=torch.zeros(4, dtype=torch.bool),
        return_count=True,
    )
    assert skipped is None and skipped_count == 0


def test_eval_vae_video_canonicalization_uses_temporal_axis():
    video = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1, 1)
    mask = torch.tensor([[False, False, True, False, True]])
    actual = _canonicalize_eval_video(video, mask)
    torch.testing.assert_close(actual.flatten(), torch.tensor([0., 1., 1., 3., 3.]))


def test_action_metrics_use_valid_step_and_dimension_denominators():
    target = torch.zeros(1, 4, 3)
    pred = torch.ones_like(target)
    l1, l2, count = masked_action_errors(
        pred,
        target,
        action_is_pad=torch.tensor([[False, True, False, True]]),
        action_dim_is_pad=torch.tensor([False, True, False]),
        return_count=True,
    )
    assert count == 4
    assert l1 == 1.0 and l2 == 1.0

    changed = pred.clone()
    changed[:, 1] = 100.0
    changed[:, :, 1] = -100.0
    changed_l1, changed_l2, changed_count = masked_action_errors(
        changed,
        target,
        action_is_pad=torch.tensor([[False, True, False, True]]),
        action_dim_is_pad=torch.tensor([False, True, False]),
        return_count=True,
    )
    assert (changed_l1, changed_l2, changed_count) == (l1, l2, count)

    skipped = masked_action_errors(
        pred,
        target,
        action_is_pad=torch.ones(4, dtype=torch.bool),
    )
    assert skipped == (None, None)


def test_eval_adapter_preserves_fixed_rng_masked_idm_training_loss():
    model = make_model(FastWAMIDM)
    sample = _unbatched_eval_sample()
    batched = EasyWAMTrainer._to_batched_eval_sample(sample)
    manually_batched = {
        key: (
            value.unsqueeze(0)
            if isinstance(value, torch.Tensor) and key != "action_horizon"
            else value
        )
        for key, value in sample.items()
    }
    manually_batched["action_horizon"] = batched["action_horizon"]
    torch.manual_seed(51)
    manual_loss, _ = model.training_loss(manually_batched)
    torch.manual_seed(51)
    adapted_loss, _ = model.training_loss(batched)
    torch.testing.assert_close(manual_loss, adapted_loss, atol=1e-6, rtol=1e-6)

    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batched.items()}
    changed["video"][:, :, 2:] += 3.0
    changed["action"][:, 2:] += 3.0

    torch.manual_seed(51)
    loss, _ = model.training_loss(batched)
    torch.manual_seed(51)
    changed_loss, _ = model.training_loss(changed)
    torch.testing.assert_close(loss, changed_loss, atol=1e-6, rtol=1e-6)


class _EvalAccelerator:
    device = torch.device("cpu")
    process_index = 0
    is_main_process = True

    def unwrap_model(self, model):
        return model

    def autocast(self):
        from contextlib import nullcontext

        return nullcontext()

    def gather_for_metrics(self, values):
        return values


def test_evaluate_tiny_idm_preserves_masked_video_count():
    model = make_model(FastWAMIDM).eval()
    raw = _unbatched_eval_sample()
    video = raw["video"]
    frames = [
        Image.fromarray(
            (((video[:, t].clamp(-1.0, 1.0) + 1.0) * 127.5)
             .byte()
             .permute(1, 2, 0)
             .numpy())
        )
        for t in range(video.shape[1])
    ]
    model.infer = lambda **kwargs: {"video": frames}

    trainer = object.__new__(EasyWAMTrainer)
    trainer.model = model
    trainer.val_dataset = [raw]
    trainer.global_step = 0
    trainer.eval_num_inference_steps = 1
    trainer.eval_save_video = False
    trainer.accelerator = _EvalAccelerator()

    torch.manual_seed(23)
    result = trainer.evaluate()
    assert result["video_valid_frames"] == 2
    assert {"psnr_rg", "ssim_rg", "psnr_rd", "ssim_rd"} <= result.keys()

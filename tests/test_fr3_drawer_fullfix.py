from collections.abc import Mapping

import pytest
import torch
from omegaconf import OmegaConf

from data.lerobot.processors.wam_processor import WAMProcessor
from data.lerobot.transforms.action_state_merger import ConcatLeftAlign
from data.lerobot.transforms.relative_action import RelativeJointTransform
from data.lerobot.utils.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _stats(dim=8, *, zero_std=False, std_value=1.0):
    std = torch.zeros(dim) if zero_std else torch.full((dim,), std_value)
    return {
        kind: {
            "default": {
                "global_mean": torch.zeros(dim),
                "global_std": std.clone(),
            }
        }
        for kind in ("action", "state")
    }


def _shape_meta(dim=8):
    return {
        "images": [{"key": "cam", "raw_shape": [1, 2, 2], "shape": [1, 2, 2]}],
        "action": [{"key": "default", "raw_shape": dim, "shape": dim}],
        "state": [{"key": "default", "raw_shape": dim, "shape": dim}],
    }


def _processor(*, state_clip_range=(-5.0, 5.0), action_clip_range=None,
               binary_dims=None, require_raw_state_anchor=False, stats=None,
               with_relative=True, num_obs_steps=1):
    processor = WAMProcessor(
        shape_meta=_shape_meta(),
        num_obs_steps=num_obs_steps,
        num_output_cameras=1,
        action_output_dim=8,
        proprio_output_dim=8,
        action_state_transforms=(
            [RelativeJointTransform(keys=["default"], joint_dim=7)] if with_relative else []
        ),
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms={"cam": []},
        val_transforms={"cam": []},
        action_clip_range=action_clip_range,
        state_clip_range=state_clip_range,
        binary_dims=binary_dims,
        require_raw_state_anchor=require_raw_state_anchor,
    )
    processor.set_normalizer_from_stats(_stats() if stats is None else stats)
    return processor


def test_new_recipe_roundtrips_drawer_joint_and_action_tails_without_clip():
    shape_meta = {kind: [{"key": "default", "shape": 8}] for kind in ("action", "state")}
    stats = _stats()
    normalizer = LinearNormalizer(
        shape_meta,
        False,
        "z-score",
        None,
        stats,
        action_clip_range=None,
        state_clip_range=None,
    )
    state = torch.tensor([[[-2.808565, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.0]]])
    action = torch.tensor([[[12.0, -11.0, 8.0, -7.0, 6.0, -9.0, 10.0, 0.25]]])
    batch = normalizer.forward({"action": {"default": action.clone()}, "state": {"default": state.clone()}})
    restored = normalizer.backward(batch)
    torch.testing.assert_close(restored["state"]["default"], state)
    torch.testing.assert_close(restored["action"]["default"], action)


def test_fixed_binary_dims_ignore_sparse_or_zero_variance_stats_and_keep_predictions_continuous():
    shape_meta = {kind: [{"key": "default", "shape": 8}] for kind in ("action", "state")}
    normalizer = LinearNormalizer(
        shape_meta,
        False,
        "z-score",
        None,
        _stats(zero_std=True),
        action_clip_range=(-5.0, 5.0),
        state_clip_range=(-5.0, 5.0),
        binary_dims={"action": {"default": [7]}, "state": {"default": [7]}},
    )
    for category in ("action", "state"):
        field = normalizer.normalizers[category]["default"]
        assert field.scale[7] == 2
        assert field.offset[7] == -1
    raw = torch.zeros(2, 3, 8)
    raw[1, :, 7] = 1
    batch = normalizer.forward({"action": {"default": raw.clone()}, "state": {"default": raw.clone()}})
    torch.testing.assert_close(batch["action"]["default"][..., 7], raw[..., 7] * 2 - 1)
    torch.testing.assert_close(batch["state"]["default"][..., 7], raw[..., 7] * 2 - 1)

    prediction = torch.zeros_like(batch["action"]["default"])
    prediction[..., 7] = torch.tensor([[-0.2, 1.4, 0.3]])
    decoded = normalizer.backward({"action": {"default": prediction}, "state": {"default": prediction.clone()}})
    torch.testing.assert_close(decoded["action"]["default"][..., 7], (prediction[..., 7] + 1) / 2)
    assert not torch.equal(decoded["action"]["default"][..., 7], decoded["action"]["default"][..., 7].round())

    bad = raw.clone()
    bad[0, 0, 7] = 0.25
    with pytest.raises(ValueError, match="0/1"):
        normalizer.forward({"action": {"default": bad}, "state": {"default": raw.clone()}})
    bad[..., 7] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        normalizer.forward({"action": {"default": bad}, "state": {"default": raw.clone()}})


@pytest.mark.parametrize(
    "binary_dims, pattern",
    [
        ({"action": {"missing": [7]}}, "Unknown binary_dims"),
        ({"action": {"default": [7, 7]}}, "duplicate"),
        ({"state": {"default": [8]}}, "outside"),
        ({"other": {"default": [7]}}, "Unknown binary_dims"),
    ],
)
def test_binary_dim_schema_fails_closed(binary_dims, pattern):
    shape_meta = {kind: [{"key": "default", "shape": 8}] for kind in ("action", "state")}
    with pytest.raises((TypeError, ValueError), match=pattern):
        LinearNormalizer(shape_meta, False, "z-score", None, _stats(), binary_dims=binary_dims)


def test_raw_anchor_overrides_clipped_proprio_for_relative_decode():
    raw_state = torch.tensor([[[-2.808565, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    raw_action = raw_state.expand(1, 2, 8).clone()
    raw_action[..., 0] += 0.75

    processor = _processor(require_raw_state_anchor=True, stats=_stats(std_value=0.1))
    payload = processor.action_state_transform({
        "action": {"default": raw_action.clone()},
        "state": {"default": raw_state.clone()},
    })
    normalized = processor.normalizer.forward(payload)
    clipped_proprio = normalized["state"]["default"]
    assert clipped_proprio[..., 0].abs().max() == 5

    with pytest.raises(ValueError, match="raw_state anchor"):
        processor.postprocess({"action": normalized["action"]["default"], "proprio": clipped_proprio})

    restored = processor.postprocess(
        {"action": normalized["action"]["default"].clone(), "proprio": clipped_proprio.clone()},
        raw_state=raw_state,
    )
    torch.testing.assert_close(restored["action"]["default"], raw_action)
    torch.testing.assert_close(restored["state"]["default"][..., 0], torch.full((1, 1), -0.5))

    legacy = _processor(require_raw_state_anchor=False, stats=_stats(std_value=0.1))
    clipped = legacy.postprocess(
        {"action": normalized["action"]["default"].clone(), "proprio": clipped_proprio.clone()}
    )
    assert not torch.allclose(clipped["action"]["default"], raw_action)


def test_raw_state_capture_is_explicit_and_does_not_change_default_sample_schema():
    processor = _processor()
    state = {"default": torch.tensor([[[1.0] * 8]])}
    captured = processor._clone_state(state)
    assert isinstance(captured, Mapping)
    assert captured["default"] is not state["default"]
    torch.testing.assert_close(captured["default"], state["default"])


def test_unbatched_preprocess_sidechannel_roundtrips_without_extra_batch_broadcast():
    processor = _processor(
        require_raw_state_anchor=True, with_relative=False, num_obs_steps=17
    )
    processor.train()
    state_horizon = 17
    action_horizon = 16
    state = torch.tensor(
        [[-2.808565, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * state_horizon
    )
    action = state[:action_horizon].clone()
    action[:, 0] += 0.75
    sample = processor.preprocess(
        {
            "images": {"cam": torch.zeros(state_horizon, 1, 2, 2)},
            "image_is_pad": torch.zeros(state_horizon, dtype=torch.bool),
            "action": {"default": action.clone()},
            "state": {"default": state.clone()},
            "action_is_pad": torch.zeros(action_horizon, dtype=torch.bool),
            "state_is_pad": torch.zeros(state_horizon, dtype=torch.bool),
            "action_dim_is_pad": torch.zeros(8, dtype=torch.bool),
            "state_dim_is_pad": torch.zeros(8, dtype=torch.bool),
            "idx": 0,
        },
        return_raw_state=True,
    )
    assert sample["action"].shape == (action_horizon, 8)
    assert sample["proprio"].shape == (state_horizon, 8)
    assert sample["raw_state"]["default"].shape == (1, 1, 8)
    assert sample["action_is_pad"].shape == (action_horizon,)
    assert sample["proprio_is_pad"].shape == (state_horizon,)
    assert sample["action_dim_is_pad"].shape == (8,)
    assert sample["proprio_dim_is_pad"].shape == (8,)

    restored = processor.postprocess(
        {"action": sample["action"], "proprio": sample["proprio"]},
        raw_state=sample["raw_state"],
    )
    assert restored["action"]["default"].shape == (1, action_horizon, 8)
    assert restored["state"]["default"].shape == (1, state_horizon, 8)
    torch.testing.assert_close(restored["action"]["default"], action.unsqueeze(0))
    torch.testing.assert_close(restored["state"]["default"], state.unsqueeze(0))


def test_raw_anchor_rejects_scalar_batch_mismatch_nan_and_invalid_binary():
    processor = _processor(
        require_raw_state_anchor=True,
        binary_dims={"state": {"default": [7]}},
        stats=_stats(std_value=0.1),
    )
    action = torch.zeros(2, 3, 8)
    proprio = torch.zeros(2, 1, 8)
    with pytest.raises(ValueError, match="scalar"):
        processor.postprocess({"action": action, "proprio": proprio}, raw_state=torch.tensor(0.0))
    with pytest.raises(ValueError, match="batch mismatch"):
        processor.postprocess(
            {"action": action, "proprio": proprio},
            raw_state={"default": torch.zeros(1, 1, 8)},
        )
    nan_state = torch.zeros(2, 1, 8)
    nan_state[..., 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        processor.postprocess({"action": action, "proprio": proprio}, raw_state=nan_state)
    invalid_binary = torch.zeros(2, 1, 8)
    invalid_binary[..., 7] = 0.5
    with pytest.raises(ValueError, match="0/1"):
        processor.postprocess(
            {"action": action, "proprio": proprio}, raw_state=invalid_binary
        )


def test_raw_anchor_aligns_cpu_input_to_action_device_and_dtype():
    processor = _processor(require_raw_state_anchor=True, stats=_stats(std_value=0.1))
    action = torch.zeros(1, 2, 8, dtype=torch.float64)
    proprio = torch.zeros(1, 1, 8, dtype=torch.float64)
    raw_state = torch.zeros(1, 1, 8, dtype=torch.float32)
    restored = processor.postprocess(
        {"action": action, "proprio": proprio}, raw_state=raw_state
    )
    assert restored["state"]["default"].dtype == torch.float64
    assert restored["state"]["default"].device == action.device


def test_processor_config_roundtrip_keeps_opt_in_keys():
    cfg = OmegaConf.create(
        {
            "state_clip_range": None,
            "action_clip_range": None,
            "binary_dims": {"action": {"default": [7]}, "state": {"default": [7]}},
            "require_raw_state_anchor": True,
        }
    )
    restored = OmegaConf.create(OmegaConf.to_yaml(cfg))
    assert restored.state_clip_range is None
    assert restored.action_clip_range is None
    assert restored.binary_dims.action.default == [7]
    assert restored.require_raw_state_anchor is True


def test_require_raw_state_anchor_requires_a_real_boolean():
    with pytest.raises(TypeError, match="boolean"):
        _processor(require_raw_state_anchor="true")

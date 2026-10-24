"""CPU contracts for FR3 absolute data and chunk-relative training targets."""

from pathlib import Path

import pytest
import torch
from data.lerobot.transforms.relative_action import RelativeJointTransform
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("batched", [False, True])
def test_joint_targets_use_current_state_and_preserve_gripper(batched):
    action = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    state = torch.arange(40, dtype=torch.float32).reshape(5, 8) + 100
    if batched:
        action = torch.stack([action, action + 50])
        state = torch.stack([state, state + 200])
    original = action.clone()
    batch = {"action": {"default": action}, "state": {"default": state}}
    transform = RelativeJointTransform(["default"], joint_dim=7)

    relative = transform.forward(batch)["action"]["default"]
    torch.testing.assert_close(
        relative[..., :7], original[..., :7] - state[..., :1, :7]
    )
    torch.testing.assert_close(relative[..., 7], original[..., 7], atol=0, rtol=0)
    torch.testing.assert_close(action, original, atol=0, rtol=0)
    restored = transform.backward(batch)["action"]["default"]
    torch.testing.assert_close(restored, original, atol=0, rtol=0)


def test_joint_transform_default_keeps_all_dimension_behavior():
    action = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    state = torch.arange(8, dtype=torch.float32).reshape(1, 8) + 2
    batch = {"action": {"default": action}, "state": {"default": state}}
    transform = RelativeJointTransform(["default"])
    torch.testing.assert_close(
        transform.forward(batch)["action"]["default"], action - state
    )
    torch.testing.assert_close(transform.backward(batch)["action"]["default"], action)
    observation = {"state": {"default": state}}
    assert transform.forward(observation) is observation


def test_fr3_recipe_preserves_camera_view_and_relative_target_statistics():
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["task=real_world_fr3_fastwam_idm_wan22"],
        )
    data = cfg.data.train
    assert cfg.model._target_ == "runtime.create_fastwam_idm"
    assert cfg.model.action_dit_config.action_dim == cfg.model.state_dim == 8
    assert data.num_frames == 17 and data.action_video_freq_ratio == 4
    assert data.global_sample_stride == 1
    assert [meta.key for meta in data.shape_meta.images] == ["left", "wrist"]
    assert list(data.video_size) == [224, 448]
    processor = instantiate(data.processor)

    # Each statistics row is a chunk, anchored to its own current state.
    state = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.25]],
            [[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.75]],
        ]
    )
    action = state.expand(-1, 16, -1).clone()
    action[..., :7] += torch.arange(16, dtype=torch.float32)[None, :, None]
    action[..., 7] = 0.5
    transformed = processor.action_state_transform(
        {"action": {"default": action}, "state": {"default": state}}
    )
    expected = torch.arange(16, dtype=torch.float32).view(1, 16, 1).expand(2, -1, 7)
    torch.testing.assert_close(transformed["action"]["default"][..., :7], expected)
    torch.testing.assert_close(
        transformed["action"]["default"][..., 7], torch.full((2, 16), 0.5)
    )
    torch.testing.assert_close(transformed["state"]["default"], state)
    assert processor.delta_action_dim_mask is None
    assert processor.norm_default_mode == "z-score"

    frames = torch.full((5, 3, 480, 640), 255, dtype=torch.uint8)
    for transform in processor.train_transforms:
        frames = transform(frames)
    assert frames.shape == (5, 3, 224, 224)
    torch.testing.assert_close(frames[..., 28:196, :], torch.ones((5, 3, 168, 224)))
    assert not frames[..., :28, :].any()
    assert not frames[..., 196:, :].any()

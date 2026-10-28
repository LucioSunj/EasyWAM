import pytest
import torch

from data.lerobot.transforms.relative_action import RelativeJointTransform


@pytest.mark.parametrize("joint_dim", [0, -1])
def test_relative_joint_rejects_nonpositive_width(joint_dim):
    with pytest.raises(ValueError, match="joint_dim must be positive"):
        RelativeJointTransform(keys=["default"], joint_dim=joint_dim)


@pytest.mark.parametrize("method", ["forward", "backward"])
@pytest.mark.parametrize("joint_dim,action_dim,state_dim", [(7, 6, 8), (7, 8, 6), (None, 8, 7)])
def test_relative_joint_rejects_width_mismatch(method, joint_dim, action_dim, state_dim):
    transform = RelativeJointTransform(keys=["default"], joint_dim=joint_dim)
    batch = {
        "action": {"default": torch.zeros(2, 32, action_dim)},
        "state": {"default": torch.zeros(2, 33, state_dim)},
    }
    with pytest.raises(ValueError, match="exceeds action/state dimensions"):
        getattr(transform, method)(batch)


@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("joint_dim", [7, None])
def test_relative_joint_roundtrip_preserves_inputs_and_absolute_tail(batch_shape, joint_dim):
    action = torch.arange(32 * 8, dtype=torch.float64).reshape(32, 8) / 100
    state = torch.arange(33 * 8, dtype=torch.float64).reshape(33, 8) / 10
    if batch_shape:
        action = action.expand(*batch_shape, 32, 8).clone()
        state = state.expand(*batch_shape, 33, 8).clone()
    original_action, original_state = action.clone(), state.clone()
    transform = RelativeJointTransform(keys=["default"], joint_dim=joint_dim)
    batch = {"action": {"default": action}, "state": {"default": state}}
    transformed = transform.forward(batch)["action"]["default"]
    width = 8 if joint_dim is None else joint_dim
    expected = original_action.clone()
    expected[..., :width] -= original_state[..., :1, :width]
    torch.testing.assert_close(transformed, expected)
    if joint_dim == 7:
        torch.testing.assert_close(transformed[..., 7], original_action[..., 7])
    torch.testing.assert_close(action, original_action)
    torch.testing.assert_close(state, original_state)

    restored = transform.backward(batch)["action"]["default"]
    torch.testing.assert_close(restored, original_action)
    torch.testing.assert_close(transformed, expected)
    torch.testing.assert_close(state, original_state)


def test_relative_joint_allows_state_only_observation():
    batch = {"state": {"default": torch.zeros(33, 8)}}
    assert RelativeJointTransform(keys=["default"], joint_dim=7).forward(batch) is batch

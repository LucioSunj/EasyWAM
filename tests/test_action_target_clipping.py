from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from data.lerobot.utils.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def field_stats():
    return {
        'mean': torch.tensor([0.2, -0.1]),
        'std': torch.tensor([0.1, 0.2]),
        'min': torch.tensor([-0.1, -0.5]),
        'max': torch.tensor([0.5, 0.3]),
        'q01': torch.tensor([-0.05, -0.4]),
        'q99': torch.tensor([0.45, 0.2]),
    }


@pytest.mark.parametrize('mode', ['z-score', 'min/max', 'q01/q99', '0/1'])
def test_default_preserves_legacy_clipping(mode):
    norm = SingleFieldLinearNormalizer(field_stats(), mode=mode)
    x = torch.tensor([[-4., 8.], [0.2, -0.1]])
    torch.testing.assert_close(norm.forward(x), (x * norm.scale + norm.offset).clamp(-5, 5))


@pytest.mark.parametrize('mode', ['z-score', 'min/max', 'q01/q99', '0/1'])
def test_disabled_clipping_preserves_extreme_targets_and_roundtrip(mode):
    norm = SingleFieldLinearNormalizer(field_stats(), mode=mode, clip_range=None)
    x = torch.tensor([[[-4., 8.], [0.2, -0.1]]])
    normalized = norm.forward(x)
    assert normalized.abs().max() > 5
    torch.testing.assert_close(norm.backward(normalized), x, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('bad', [[], [1], [1, 2, 3], [1, 1], [2, 1], [float('nan'), 2], [-2, float('inf')], 'bad'])
def test_invalid_clip_range_is_rejected(bad):
    with pytest.raises((TypeError, ValueError), match='clip_range'):
        SingleFieldLinearNormalizer(field_stats(), mode='z-score', clip_range=bad)


def test_custom_clip_range():
    norm = SingleFieldLinearNormalizer(field_stats(), mode='z-score', clip_range=[-2, 3])
    torch.testing.assert_close(norm.forward(torch.tensor([[-9., 9.]])), torch.tensor([[-2., 3.]]))


@pytest.mark.parametrize('stepwise', [False, True])
def test_action_only_opt_out_keeps_state_behavior(stepwise):
    shape_meta = {kind: [{'key': 'default', 'shape': 2}] for kind in ('action', 'state')}
    stats = {kind: {'default': {f'{prefix}_{key}': value.clone() for prefix in ('global', 'stepwise') for key, value in field_stats().items()}} for kind in ('action', 'state')}
    norm = LinearNormalizer(shape_meta, stepwise, 'z-score', None, stats, action_clip_range=None)
    x = torch.tensor([[[-4., 8.]]])
    batch = norm.forward({'action': {'default': x.clone()}, 'state': {'default': x.clone()}})
    assert batch['action']['default'].abs().max() > 5
    assert batch['state']['default'].abs().max() == 5
    torch.testing.assert_close(norm.normalizers['action']['default'].backward(batch['action']['default']), x)


def test_processor_hydra_config_records_opt_out_and_preserves_gripper():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=['task=real_world_fr3_fastwam_idm_wan22'])
    OmegaConf.set_struct(cfg, False)
    cfg.data.train.processor.action_clip_range = None
    processor = instantiate(cfg.data.train.processor)
    stats = {kind: {'default': {'global_mean': torch.zeros(8), 'global_std': torch.full((8,), .1)}} for kind in ('action', 'state')}
    stats['action']['default']['global_mean'][7] = .5
    stats['action']['default']['global_std'][7] = .5
    processor.set_normalizer_from_stats(stats)
    state = torch.zeros(2, 1, 8)
    action = torch.ones(2, 16, 8)
    action[..., 7] = torch.arange(16) % 2
    batch = processor.action_state_transform({'action': {'default': action.clone()}, 'state': {'default': state.clone()}})
    batch = processor.normalizer.forward(batch)
    assert batch['action']['default'][..., :7].max() > 5
    torch.testing.assert_close(batch['action']['default'][..., 7], 2 * action[..., 7] - 1)
    restored = processor.postprocess({'action': batch['action']['default'], 'proprio': batch['state']['default']})
    torch.testing.assert_close(restored['action']['default'], action)
    reloaded = OmegaConf.create(OmegaConf.to_yaml(cfg))
    assert 'action_clip_range' in reloaded.data.train.processor
    assert reloaded.data.train.processor.action_clip_range is None

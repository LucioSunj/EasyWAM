from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from data.lerobot.robot_video_dataset import RobotVideoDataset


def test_fr3_postprocess_retains_full_action_chunk_and_inverts_transforms():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=['task=real_world_fr3_fastwam_idm_wan22'])
    processor = instantiate(cfg.data.train.processor)
    stats = {kind: {'default': {'global_mean': torch.full((8,), .3),
                              'global_std': torch.full((8,), .7)}} for kind in ['action', 'state']}
    processor.set_normalizer_from_stats(stats)
    state = torch.full((2, 1, 8), .2)
    action = torch.full((2, 16, 8), .4)
    action[..., 7] = .8
    payload = processor.action_state_transform({'action': {'default': action.clone()},
                                                'state': {'default': state.clone()}})
    payload = processor.normalizer.forward(payload)
    restored = processor.postprocess({'action': payload['action']['default'],
                                      'proprio': payload['state']['default']})
    assert restored['action']['default'].shape == (2, 16, 8)
    torch.testing.assert_close(restored['action']['default'], action)
    torch.testing.assert_close(restored['state']['default'], state)


def test_returned_proprio_mask_matches_trimmed_horizon():
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.max_padding_retry = 0
    dataset.skip_padding_as_possible = False
    dataset.video_sample_indices = (0, 4, 8, 12, 16)
    dataset.concat_multi_camera = 'horizontal'
    dataset.override_instruction = None
    dataset.resize_transform = dataset.crop_transform = dataset.normalize_transform = lambda x: x
    dataset._get_cached_text_context = lambda prompt: (torch.zeros(128, 4096), torch.ones(128, dtype=torch.bool))
    dataset.lerobot_dataset = [{'pixel_values': torch.zeros(2, 5, 3, 224, 224),
        'action': torch.zeros(16, 8), 'proprio': torch.zeros(17, 8), 'instruction': 'test',
        'image_is_pad': torch.zeros(5, dtype=torch.bool), 'action_is_pad': torch.zeros(16, dtype=torch.bool),
        'proprio_is_pad': torch.arange(17) > 4}]
    sample = dataset._get(0)
    assert sample['proprio'].shape[0] == sample['proprio_is_pad'].shape[0] == 16

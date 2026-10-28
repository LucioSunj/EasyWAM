"""Retention policy and failed-save safety without loading a training model."""
from pathlib import Path
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from trainer import EasyWAMTrainer


class InitStopped(Exception):
    pass


def init_limits(overrides):
    cfg = OmegaConf.create(dict(output_dir='unused', learning_rate=1e-4, warmup_ratio=.05,
        weight_decay=.01, batch_size=8, num_workers=0, max_steps=20000,
        log_every=100, save_every=2000, eval_every=1000, eval_num_inference_steps=10,
        gradient_accumulation_steps=4, max_grad_norm=1., seed=42, resume=None,
        mixed_precision='bf16', wandb={'enabled': False}, **overrides))

    class StopAccelerator:
        @property
        def distributed_type(self):
            raise InitStopped

    trainer = EasyWAMTrainer.__new__(EasyWAMTrainer)
    with pytest.raises(InitStopped):
        trainer.__init__(None, None, cfg=cfg, accelerator=StopAccelerator())
    return trainer


@pytest.mark.parametrize('overrides,expected', [({}, (5, 5)),
    ({'checkpoint_save_limit': 3}, (3, 3)),
    ({'checkpoint_save_limit': None}, (None, None)),
    ({'checkpoint_save_limit': 3, 'state_checkpoint_save_limit': 1}, (3, 1)),
    ({'checkpoint_save_limit': None, 'state_checkpoint_save_limit': 1}, (None, 1)),
    ({'checkpoint_save_limit': 3, 'state_checkpoint_save_limit': None}, (3, None))])
def test_limits_defaults_and_independence(overrides, expected):
    trainer = init_limits(overrides)
    assert (trainer.checkpoint_save_limit, trainer.state_checkpoint_save_limit) == expected


@pytest.mark.parametrize('value', [0, -1])
def test_invalid_state_limit(value):
    with pytest.raises(ValueError, match='state_checkpoint_save_limit'):
        init_limits({'state_checkpoint_save_limit': value})


def fixture_trainer(tmp_path, weight_limit=3, state_limit=1):
    trainer = EasyWAMTrainer.__new__(EasyWAMTrainer)
    trainer.weights_dir = str(tmp_path / 'weights')
    trainer.state_dir = str(tmp_path / 'state')
    trainer.checkpoint_save_limit = weight_limit
    trainer.state_checkpoint_save_limit = state_limit
    for directory in [Path(trainer.weights_dir), Path(trainer.state_dir)]:
        directory.mkdir()
    for step in (2000, 4000, 6000, 8000):
        (Path(trainer.weights_dir) / f'step_{step:06d}.pt').write_text('weights')
        state = Path(trainer.state_dir) / f'step_{step:06d}'
        state.mkdir()
        (state / 'trainer_state.json').write_text('{}')
    return trainer


@pytest.mark.parametrize('weight_limit,state_limit,counts', [(3, 1, (3, 1)), (None, 1, (4, 1)), (3, None, (3, 4))])
def test_rolling_prune_is_scoped(tmp_path, weight_limit, state_limit, counts):
    trainer = fixture_trainer(tmp_path, weight_limit, state_limit)
    weights, states = Path(trainer.weights_dir), Path(trainer.state_dir)
    (weights / 'step_010000.pt.tmp').write_text('unfinished')
    (states / 'notes').mkdir()
    (states / 'step_000001').symlink_to(states / 'notes', target_is_directory=True)
    trainer._prune_saved_checkpoints()
    assert len(list(weights.glob('*.pt'))) == counts[0]
    actual_states = [p for p in states.glob('step_*') if not p.is_symlink()]
    assert len(actual_states) == counts[1]
    assert (weights / 'step_008000.pt').exists() and (states / 'step_008000').exists()
    assert (weights / 'step_010000.pt.tmp').exists() and (states / 'notes').exists()
    assert (states / 'step_000001').is_symlink()


@pytest.mark.parametrize('save_fails', [False, True])
def test_old_state_survives_until_new_save_completes(tmp_path, save_fails):
    trainer = fixture_trainer(tmp_path)
    trainer.global_step = 10000
    trainer.batch_size = 8
    trainer.epoch = trainer.batch_in_epoch = 0

    def save_weights(*, step_tag):
        path = Path(trainer.weights_dir) / (step_tag + '.pt')
        path.write_text('new weights')
        return str(path)

    def save_state(*, output_dir):
        assert (Path(trainer.state_dir) / 'step_008000').exists()
        if save_fails:
            raise OSError('synthetic incomplete state')
        (Path(output_dir) / 'shard.pt').write_text('new state')

    trainer._save_weights_checkpoint = save_weights
    trainer.accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=lambda: None,
        save_state=save_state, num_processes=2, gradient_accumulation_steps=4)
    if save_fails:
        with pytest.raises(OSError, match='incomplete'):
            trainer.save_checkpoint()
        assert (Path(trainer.state_dir) / 'step_008000').exists()
        assert not (Path(trainer.state_dir) / 'step_010000/trainer_state.json').exists()
    else:
        trainer.save_checkpoint()
        assert sorted(p.name for p in Path(trainer.state_dir).iterdir()) == ['step_010000']
        assert (Path(trainer.state_dir) / 'step_010000/trainer_state.json').exists()
        assert len(list(Path(trainer.weights_dir).glob('*.pt'))) == 3


def test_completed_state_rejects_overwrite_before_any_writes(tmp_path):
    trainer = fixture_trainer(tmp_path)
    trainer.global_step = 8000
    trainer.batch_size = 8
    trainer.epoch = trainer.batch_in_epoch = 0
    state_path = Path(trainer.state_dir) / 'step_008000'
    old_marker = (state_path / 'trainer_state.json').read_text()
    (state_path / 'shard.pt').write_text('old shard')
    calls = []
    trainer._save_weights_checkpoint = lambda **kwargs: calls.append('weights')
    trainer.accelerator = SimpleNamespace(
        is_main_process=True,
        wait_for_everyone=lambda: calls.append('barrier'),
        save_state=lambda **kwargs: calls.append('state'),
        num_processes=2,
        gradient_accumulation_steps=4,
    )

    with pytest.raises(FileExistsError, match='completed checkpoint state'):
        trainer.save_checkpoint()

    assert calls == []
    assert (state_path / 'trainer_state.json').read_text() == old_marker
    assert (state_path / 'shard.pt').read_text() == 'old shard'


def test_partial_state_failure_leaves_no_completion_marker(tmp_path):
    trainer = fixture_trainer(tmp_path)
    trainer.global_step = 10000
    trainer.batch_size = 8
    trainer.epoch = trainer.batch_in_epoch = 0

    def save_weights(*, step_tag):
        path = Path(trainer.weights_dir) / (step_tag + '.pt')
        path.write_text('new weights')
        return str(path)

    def save_state(*, output_dir):
        (Path(output_dir) / 'partial-shard.pt').write_text('partial')
        raise OSError('synthetic incomplete state')

    trainer._save_weights_checkpoint = save_weights
    trainer.accelerator = SimpleNamespace(
        is_main_process=True,
        wait_for_everyone=lambda: None,
        save_state=save_state,
        num_processes=2,
        gradient_accumulation_steps=4,
    )
    with pytest.raises(OSError, match='incomplete'):
        trainer.save_checkpoint()

    failed_state = Path(trainer.state_dir) / 'step_010000'
    assert (failed_state / 'partial-shard.pt').exists()
    assert not (failed_state / 'trainer_state.json').exists()


def test_other_rank_completed_marker_rejects_before_any_writes(tmp_path):
    trainer = EasyWAMTrainer.__new__(EasyWAMTrainer)
    trainer.state_dir = str(tmp_path / 'state')
    trainer.weights_dir = str(tmp_path / 'weights')
    trainer.global_step = 10000
    calls = []
    trainer._save_weights_checkpoint = lambda **kwargs: calls.append('weights')

    class CollectiveAccelerator:
        is_main_process = True
        device = torch.device('cpu')

        def gather(self, local_marker):
            # This rank sees no marker locally; another rank has one.
            return torch.tensor([0, 1], device=local_marker.device)

        def wait_for_everyone(self):
            calls.append('barrier')

        def save_state(self, **kwargs):
            calls.append('state')

    trainer.accelerator = CollectiveAccelerator()
    with pytest.raises(FileExistsError, match='completed checkpoint state'):
        trainer.save_checkpoint()
    assert calls == []
    assert not (Path(trainer.state_dir) / 'step_010000').exists()


def test_resume_at_max_steps_does_not_iterate_or_save(tmp_path):
    trainer = EasyWAMTrainer.__new__(EasyWAMTrainer)
    trainer.max_steps = 1
    trainer.global_step = 1
    trainer._resume_state_loaded = True
    events = []
    trainer._set_dit_only_train_mode = lambda: events.append('train_mode')
    trainer.save_checkpoint = lambda: pytest.fail('already-complete resume must not save')

    trainer.train()
    assert events == ['train_mode']


@pytest.mark.parametrize('field,value', [('batch_size', 4), ('num_processes', 4),
                                       ('gradient_accumulation_steps', 2)])
def test_resume_rejects_changed_batch_contract_before_loading(tmp_path, field, value):
    trainer = fixture_trainer(tmp_path)
    trainer.batch_size = 8
    trainer.accelerator = SimpleNamespace(num_processes=2, gradient_accumulation_steps=4,
        load_state=lambda **kwargs: pytest.fail('mismatched state must not be loaded'))
    payload = dict(global_step=10000, epoch=0, batch_in_epoch=1, batch_size=8,
                   num_processes=2, gradient_accumulation_steps=4)
    payload[field] = value
    state = Path(trainer.state_dir) / 'step_010000'
    state.mkdir()
    (state / 'trainer_state.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=field):
        trainer.load_training_state(str(state))

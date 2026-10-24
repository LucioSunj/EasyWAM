import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainer import EasyWAMTrainer


@pytest.mark.parametrize('payload', [None, {}, {'global_step': 10}])
def test_incomplete_state_is_rejected_before_loading(tmp_path, payload):
    trainer = EasyWAMTrainer.__new__(EasyWAMTrainer)
    trainer.accelerator = SimpleNamespace(load_state=lambda **kwargs: pytest.fail('must not load'))
    if payload is not None:
        (tmp_path / 'trainer_state.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Incomplete training state'):
        trainer.load_training_state(str(tmp_path))


def test_progress_marker_is_atomic_and_published_after_all_shards(tmp_path):
    trainer = EasyWAMTrainer.__new__(EasyWAMTrainer)
    trainer.global_step = 10
    trainer.epoch = 1
    trainer.batch_in_epoch = 2
    trainer.batch_size = 8
    trainer.state_dir = str(tmp_path)
    trainer._save_weights_checkpoint = lambda **kw: 'weights.pt'
    trainer._prune_saved_checkpoints = lambda: None
    state_path = tmp_path / 'step_000010'
    events = []

    def barrier():
        events.append('barrier')

    def save_state(**kwargs):
        events.append('shards')

    original = trainer._save_trainer_state

    def save_progress(path):
        assert events[-2:] == ['shards', 'barrier']
        original(path)
        assert not Path(path, 'trainer_state.json.tmp').exists()
        events.append('progress')

    trainer._save_trainer_state = save_progress
    trainer.accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=barrier,
        save_state=save_state, num_processes=4, gradient_accumulation_steps=2)
    trainer.save_checkpoint()
    assert json.loads((state_path / 'trainer_state.json').read_text())['global_step'] == 10


def test_resume_rejects_legacy_deepspeed_step_mismatch(tmp_path):
    trainer = EasyWAMTrainer.__new__(EasyWAMTrainer)
    trainer.is_deepspeed = True
    trainer.batch_size = 8
    trainer.accelerator = SimpleNamespace(num_processes=4, gradient_accumulation_steps=2,
        load_state=lambda **kwargs: None,
        deepspeed_engine_wrapped=SimpleNamespace(engine=SimpleNamespace(global_steps=9)))
    (tmp_path / 'trainer_state.json').write_text(json.dumps({'global_step': 10, 'epoch': 1,
        'batch_in_epoch': 2, 'batch_size': 8, 'num_processes': 4, 'gradient_accumulation_steps': 2}))
    with pytest.raises(ValueError, match='inconsistent trainer and DeepSpeed'):
        trainer.load_training_state(str(tmp_path))

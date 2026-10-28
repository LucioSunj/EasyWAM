import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from fastwam_fixtures import make_model
from model.fastwam import FastWAM
from trainer import DistributedType, EasyWAMTrainer


def _constructor_cfg(output_dir, resume):
    return OmegaConf.create(
        dict(
            output_dir=str(output_dir),
            learning_rate=1e-4,
            warmup_ratio=0.0,
            weight_decay=0.01,
            batch_size=2,
            num_workers=0,
            max_steps=2,
            log_every=1,
            save_every=2,
            eval_every=0,
            eval_num_inference_steps=2,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
            resume=resume,
            mixed_precision="no",
            lr_scheduler_type="constant",
            eval_save_video=False,
            wandb={"enabled": False},
        )
    )


class _ConstructorAccelerator:
    distributed_type = DistributedType.NO
    mixed_precision = "no"
    device = torch.device("cpu")
    num_processes = 1
    process_index = 0
    is_main_process = True
    is_local_main_process = True
    gradient_accumulation_steps = 1

    def __init__(self, events, probe):
        self.events = events
        self.probe = probe
        self.state = SimpleNamespace(deepspeed_plugin=None)
        self.master_snapshot = None

    def gather(self, value):
        return value

    def prepare(self, model, optimizer, train_loader, scheduler):
        self.events.append("prepare")
        self.master_snapshot = self.probe.detach().float().clone()
        return model, optimizer, train_loader, scheduler

    def load_state(self, **kwargs):
        self.events.append("load_state")

    def wait_for_everyone(self):
        self.events.append("barrier")


def test_constructor_loads_weight_resume_before_prepare_and_only_once(tmp_path, monkeypatch):
    weight_path = tmp_path / "weights.pt"
    weight_path.write_text("weights")
    events = []
    model = make_model(FastWAM)
    probe = next(model.parameters())
    load_count = 0

    # Keep this constructor-order regression focused on the resume/prepare
    # contract rather than the model's production freeze policy.
    monkeypatch.setattr(
        EasyWAMTrainer,
        "_apply_dit_only_train_mode",
        staticmethod(lambda model: None),
    )

    def load_weights(path, optimizer=None):
        nonlocal load_count
        load_count += 1
        events.append("weights")
        with torch.no_grad():
            probe.fill_(0.25)

    monkeypatch.setattr(model, "load_checkpoint", load_weights)
    accelerator = _ConstructorAccelerator(events, probe)
    EasyWAMTrainer(
        model,
        [0],
        cfg=_constructor_cfg(tmp_path, str(weight_path)),
        accelerator=accelerator,
    )

    assert events[:2] == ["weights", "prepare"]
    assert load_count == 1
    torch.testing.assert_close(accelerator.master_snapshot, torch.full_like(probe, 0.25))

    # CPU-only approximation of the ZeRO master copy-back boundary: a stale
    # model value must be replaced by the master snapshot captured at prepare.
    with torch.no_grad():
        probe.zero_()
        probe.copy_(accelerator.master_snapshot)
    torch.testing.assert_close(probe, torch.full_like(probe, 0.25))


def test_full_state_resume_loads_after_prepare_and_null_resume_is_noop(tmp_path, monkeypatch):
    state_path = tmp_path / "state"
    state_path.mkdir()
    (state_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "epoch": 0,
                "batch_in_epoch": 0,
                "batch_size": 2,
                "num_processes": 1,
                "gradient_accumulation_steps": 1,
            }
        )
    )
    events = []
    model = make_model(FastWAM)
    probe = next(model.parameters())
    weight_load_count = 0

    monkeypatch.setattr(
        EasyWAMTrainer,
        "_apply_dit_only_train_mode",
        staticmethod(lambda model: None),
    )

    def load_weights(path, optimizer=None):
        nonlocal weight_load_count
        weight_load_count += 1

    monkeypatch.setattr(model, "load_checkpoint", load_weights)
    accelerator = _ConstructorAccelerator(events, probe)
    trainer = EasyWAMTrainer(
        model,
        [0],
        cfg=_constructor_cfg(tmp_path, str(state_path)),
        accelerator=accelerator,
    )

    assert events[:2] == ["prepare", "load_state"]
    assert trainer.global_step == 1
    assert weight_load_count == 0

    # A fresh run with resume=null must not invoke either resume loader.
    model = make_model(FastWAM)
    probe = next(model.parameters())
    accelerator = _ConstructorAccelerator([], probe)
    weight_load_count = 0
    monkeypatch.setattr(model, "load_checkpoint", load_weights)
    EasyWAMTrainer(
        model,
        [0],
        cfg=_constructor_cfg(tmp_path / "fresh", None),
        accelerator=accelerator,
    )
    assert weight_load_count == 0


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

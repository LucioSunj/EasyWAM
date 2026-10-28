"""Four-rank correctness, restore, and steady-update measurements."""
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import time

from insert_common import (ARMS, HERE, NCCL_CUMEM_HOST_ENABLE, ROOT, SEALED, STOP,
                           check_headroom, digest, distributed_save_gate, global_slots,
                           read, require_nccl_environment, stress_kind, write_new)

require_nccl_environment()

import numpy as np
import torch
from omegaconf import OmegaConf
import runtime
import trainer
from data.lerobot.robot_video_dataset import RobotVideoDataset
import data.lerobot.base_lerobot_dataset as base_dataset

ARM = os.environ['INSERT_ARM']
MODE = os.environ['INSERT_MODE']
SPEC = ARMS[ARM]
ARM_DIR = HERE / ARM
CONFIG = ARM_DIR / 'train.json'


def rng_summary():
    raw = [pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()),
           torch.random.get_rng_state().numpy().tobytes()]
    raw.extend(state.numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    return [hashlib.sha256(value).hexdigest() for value in raw]


def finite_and_sample(tensor):
    flat = tensor.detach().reshape(-1)
    finite = torch.ones((), device=tensor.device, dtype=torch.bool)
    for start in range(0, flat.numel(), 8 * 1024**2):
        finite &= torch.isfinite(flat[start:start + 8 * 1024**2]).all()
    assert bool(finite), 'Nonfinite optimizer state/master'
    offsets = sorted(set([0, max(0, flat.numel() // 2 - 8), max(0, flat.numel() - 16)]))
    sample = torch.cat([flat[i:i + 16] for i in offsets]).float().cpu().numpy().tobytes()
    return {'numel': flat.numel(), 'dtype': str(tensor.dtype),
            'sample_sha256': hashlib.sha256(sample).hexdigest()}


def optimizer_summary(engine):
    optimizer = engine.optimizer.optimizer
    result = []
    storages = {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            item = {'master': finite_and_sample(parameter), 'state': {}}
            state = optimizer.state.get(parameter, {})
            for name, value in state.items():
                if torch.is_tensor(value):
                    item['state'][name] = finite_and_sample(value)
                    storage = value.untyped_storage()
                    storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
                else:
                    item['state'][name] = value
            assert 'exp_avg' in state and 'exp_avg_sq' in state
            result.append(item)
    return {'partitions': result, 'unique_state_storage_bytes': sum(storages.values())}


def indexed_getitem(self, index):
    # Do not silently replace a failed sample with a random one in an acceptance run.
    sample = dict(self._get(index))
    sample['_comparison_index'] = int(index)
    return sample


RobotVideoDataset.__getitem__ = indexed_getitem
base_dataset.MAX_GETITEM_ATTEMPT = 1


class TimedIterator:
    def __init__(self, parent, iterator):
        self.parent, self.iterator = parent, iterator

    def __iter__(self):
        return self

    def __next__(self):
        p = self.parent
        if p.micro_in_update == 0:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            p.update_started = time.perf_counter()
            p.allocated_before = torch.cuda.memory_allocated()
        return next(self.iterator)


class TimedLoader:
    def __init__(self, parent, loader):
        self.parent, self.loader = parent, loader

    def __iter__(self):
        return TimedIterator(self.parent, iter(self.loader))

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class ComparisonTrainer(trainer.EasyWAMTrainer):
    def __init__(self, *args, **kwargs):
        self.micro_in_update = 0
        self.measurements = []
        self.sample_ids = []
        self.pending_indices = []
        self.grad_health = {}
        self.loss_health = []
        self.resume_verified = False
        self.checkpoint_calls = 0
        self.phase4 = None
        self.save_peaks = []
        self.seen_stress = set()
        super().__init__(*args, **kwargs)
        self.rank = self.accelerator.process_index
        assert self.accelerator.num_processes == 4
        assert (self.batch_size, self.gradient_accumulation_steps) == (SPEC['batch'], SPEC['gas'])
        assert self.model.zero_optimization_stage() == SPEC['zero']
        assert self.model.train_batch_size() == 64
        assert str(self.model.optimizer.gradient_accumulation_dtype) == 'torch.bfloat16'
        self.start_step = self.global_step
        assert self.start_step == (0 if MODE == 'fresh' else 3)
        assert MODE == 'fresh' or self.resume_verified
        self.max_steps = 3 if MODE == 'fresh' else STOP
        model = self.accelerator.unwrap_model(self.model)
        processor = self.train_dataset.lerobot_dataset.processor
        assert processor.action_clip_range is None and processor.state_clip_range is None
        # The audit is intentionally reused from the sealed old preparation;
        # the successor root contains only its path/hash binding, not a copied
        # large-input audit artifact.
        sealed = read(SEALED)
        audit = read(Path(sealed['old_cpu_audit_path']))
        indices = [audit['worst_action']['global_index'], audit['worst_state']['global_index'],
                   audit['worst_action']['tail_index']]
        self.stress = [self.train_dataset[index] for index in indices]
        for sample in self.stress:
            assert sample['video'].shape == (3, 9, 224, 448)
            assert sample['action'].shape == sample['proprio'].shape == (32, 8)
        assert int(self.stress[2]['action_is_pad'].sum()) == 31
        self.extremes = [float(self.stress[0]['action'][:, :7].abs().max()),
                         float(self.stress[1]['proprio'][0, :7].abs().max())]
        assert math.isclose(self.extremes[0], audit['worst_action']['max_abs_target'], rel_tol=1e-5)
        assert math.isclose(self.extremes[1], audit['worst_state']['max_abs_state'], rel_tol=1e-5)
        for name, parameter in model.named_parameters():
            if name.startswith(('state_encoder.', 'action_expert.head.')):
                parameter.register_hook(self.capture_gradient(name))
        model.register_forward_pre_hook(self.prepare_sample)
        original_checkpoint = torch.utils.checkpoint.checkpoint

        def counted_checkpoint(*args, **kwargs):
            assert kwargs.get('use_reentrant') is False
            self.checkpoint_calls += 1
            return original_checkpoint(*args, **kwargs)

        torch.utils.checkpoint.checkpoint = counted_checkpoint
        self.train_loader = TimedLoader(self, self.train_loader)

    def capture_gradient(self, name):
        def capture(value):
            if self.global_step < 4:
                finite, nonzero = torch.isfinite(value).all(), value.ne(0).any()
                old = self.grad_health.get(name)
                self.grad_health[name] = (finite if old is None else old[0] & finite,
                                          nonzero if old is None else old[1] | nonzero)
            return value
        return capture

    def _deepspeed_grad_norm(self, loss):
        value = super()._deepspeed_grad_norm(loss)
        self.latest_grad_norm = value.detach()
        return value

    def prepare_sample(self, module, args):
        original = args[0]
        slots = global_slots(SPEC['batch'], SPEC['gas'], self.rank, self.micro_in_update)
        self.pending_indices.append((slots, original['_comparison_index'].detach()))
        self.micro_in_update += 1
        if self.global_step >= 4:
            return None
        self.seen_stress.update(stress_kind(slot, self.global_step + 1) for slot in slots
                                if stress_kind(slot, self.global_step + 1) is not None)
        sample = dict(original)
        for key, value in original.items():
            if not torch.is_tensor(value) or key.startswith('_comparison'):
                continue
            sample[key] = value.clone()
            for row, slot in enumerate(slots):
                kind = stress_kind(slot, self.global_step + 1)
                if kind is not None and key in self.stress[kind] and torch.is_tensor(self.stress[kind][key]):
                    sample[key][row].copy_(self.stress[kind][key].to(sample[key]))
        return (sample,)

    def _accumulate_train_interval(self, loss, loss_dict, *args):
        self.loss_health.append(torch.stack([torch.isfinite(loss.detach()),
                                           *[torch.isfinite(v).all() for v in loss_dict.values()]]).all())
        boundary = self.accelerator.sync_gradients
        assert boundary == (self.micro_in_update == SPEC['gas'])
        if boundary:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - self.update_started
            peak = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            free, total = torch.cuda.mem_get_info()
            assert bool(torch.stack(self.loss_health).all()), 'Nonfinite loss'
            grad_norm = float(self.latest_grad_norm.cpu())
            assert math.isfinite(grad_norm), 'Nonfinite global gradient norm'
            self.loss_health.clear()
            step = self.global_step + 1
            assert self.model.global_steps == step
            ids = [(slot, int(index)) for slots, indices in self.pending_indices
                   for slot, index in zip(slots, indices.cpu().tolist())]
            self.sample_ids.append({'step': step, 'indices': ids, 'stress': step <= 4})
            self.pending_indices.clear()
            result = {'step': step, 'seconds': elapsed, 'peak_allocated': peak,
                      'peak_reserved': reserved, 'free_after': free, 'total': total,
                      'allocated_before': self.allocated_before,
                      'allocated_after': torch.cuda.memory_allocated(),
                      'microbatches': self.micro_in_update,
                      'engine_micro_steps': self.model.micro_steps}
            result.update(check_headroom(total, peak, f'{MODE} measurement step {step}'))
            result['global_grad_norm'] = grad_norm
            self.measurements.append(result)
            with (ARM_DIR / MODE / f'updates-rank-{self.rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            self.micro_in_update = 0
            if step == 4:
                self.phase4 = {'trainer_next_step': step, 'engine_step': self.model.global_steps,
                               'resume_verified': self.resume_verified, 'rank': self.rank,
                               'pid': os.getpid(), 'output_dir': str(ARM_DIR / MODE),
                               'config_sha256': digest(CONFIG),
                               'optimizer': optimizer_summary(self.model)}
                write_new(ARM_DIR / MODE / f'restored-step4-rank-{self.rank}.json', self.phase4)
        super()._accumulate_train_interval(loss, loss_dict, *args)

    def state_evidence(self):
        return {'arm': ARM, 'recipe_sha256': digest(CONFIG), 'zero': SPEC['zero'],
                'ac': SPEC['ac'], 'rng': rng_summary(), 'optimizer': optimizer_summary(self.model),
                'scheduler': self.scheduler.state_dict(), 'global_step': self.global_step,
                'engine_step': self.model.global_steps, 'epoch': self.epoch,
                'batch_in_epoch': self.batch_in_epoch,
                'accelerate_step': self.accelerator.step}

    def save_checkpoint(self):
        distributed_save_gate(self.accelerator, MODE, self.global_step)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = super().save_checkpoint()
        write_new(Path(result['state_path']) / f'comparison-rank-{self.accelerator.process_index}.json',
                  self.state_evidence())
        torch.cuda.synchronize()
        total = torch.cuda.mem_get_info()[1]
        peak = torch.cuda.max_memory_allocated()
        self.save_peaks.append({'step': self.global_step,
                                **check_headroom(total, peak, f'{MODE} save step {self.global_step}')})
        self.accelerator.wait_for_everyone()
        return result

    def load_training_state(self, state_dir):
        expected = read(Path(state_dir) / f'comparison-rank-{self.accelerator.process_index}.json')
        assert expected['arm'] == ARM and expected['recipe_sha256'] == digest(CONFIG)
        assert expected['zero'] == SPEC['zero'] and expected['ac'] == SPEC['ac']
        super().load_training_state(state_dir)
        actual = self.state_evidence()
        assert actual == expected, 'Full training/RNG/optimizer/scheduler restore mismatch'
        self.resume_verified = True

    def train(self):
        super().train()
        assert self.global_step == self.model.global_steps == self.max_steps
        if MODE == 'fresh':
            assert self.seen_stress == {0, 1, 2}
        gradients = {name: {'finite': bool(v[0]), 'nonzero': bool(v[1])}
                     for name, v in self.grad_health.items()}
        assert gradients and all(v['finite'] and v['nonzero'] for v in gradients.values())
        expected_calls = (self.max_steps - self.start_step) * SPEC['gas'] * 30 if SPEC['ac'] else 0
        assert self.checkpoint_calls == expected_calls, (self.checkpoint_calls, expected_calls)
        evaluation = None
        evaluation_peak = None
        if MODE == 'resume':
            assert self.phase4 and self.resume_verified
            torch.cuda.reset_peak_memory_stats()
            evaluation = self.evaluate()
            assert evaluation and all(math.isfinite(v) for v in evaluation.values() if isinstance(v, (int, float)))
            torch.cuda.synchronize()
            total = torch.cuda.mem_get_info()[1]
            peak = torch.cuda.max_memory_allocated()
            evaluation_peak = check_headroom(total, peak, f'{MODE} evaluation step {self.global_step}')
        result = {'arm': ARM, 'mode': MODE, 'rank': self.rank, 'spec': SPEC,
                  'config_sha256': digest(CONFIG), 'start_step': self.start_step,
                  'final_step': self.global_step, 'global_batch': 64,
                  'output_dir': str(ARM_DIR / MODE), 'pid': os.getpid(),
                  'runtime_env': {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE},
                  'grad_accum_dtype': str(self.model.optimizer.gradient_accumulation_dtype),
                  'resume_verified': self.resume_verified, 'gradients': gradients,
                  'extremes': self.extremes, 'evaluation': evaluation,
                  'seen_stress': sorted(self.seen_stress),
                  'save_peaks': self.save_peaks, 'evaluation_peak': evaluation_peak,
                  'checkpoint_calls': self.checkpoint_calls, 'measurements': self.measurements,
                  'sample_ids': self.sample_ids, 'scheduler_total_steps': 20000}
        write_new(ARM_DIR / f'acceptance-{MODE}-rank-{self.rank}.json', result)
        self.accelerator.wait_for_everyone()


cfg = OmegaConf.load(CONFIG)
cfg.output_dir = str(ARM_DIR / MODE)
cfg.log_every = 1
cfg.eval_every = 0
if MODE == 'resume':
    cfg.resume = str(ARM_DIR / 'fresh/checkpoints/state/step_000003')
trainer.EasyWAMTrainer = ComparisonTrainer
runtime.run_training(cfg)

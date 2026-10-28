"""Fresh WanRobot formal Insert training with distributed runtime proofs."""
import math
import os
import json

from insert_common import (AUTHORIZATION, CONFIG, FORMAL_OUTPUT, NCCL_CUMEM_HOST_ENABLE,
                           ROOT, check_headroom, digest, distributed_save_gate, read, require,
                           now, require_nccl_environment, validate_config, write_new)

require_nccl_environment()

import torch
from omegaconf import OmegaConf
import runtime
import trainer


class FormalTrainer(trainer.EasyWAMTrainer):
    """Keep the formal run's all-rank update/save/headroom evidence local."""

    def __init__(self, *args, **kwargs):
        self.rank = None
        self.start_step = None
        self.micro_in_update = 0
        self.loss_health = []
        self.measurements = []
        self.save_peaks = []
        self.evaluation_peaks = []
        self.latest_grad_norm = None
        super().__init__(*args, **kwargs)
        self.rank = self.accelerator.process_index
        self.start_step = self.global_step
        require(self.start_step == 0 and self.model.global_steps == 0,
                'Formal Insert must start WanRobot/DeepSpeed at step zero')
        require(self.max_steps == self.cfg.max_steps == 20000,
                'Formal Insert must retain the absolute 20000-step horizon')
        require(self.accelerator.num_processes == 4 and
                (self.batch_size, self.gradient_accumulation_steps) == (4, 4),
                'Formal Insert requires world4/B4/GAS4')
        require(self.model.zero_optimization_stage() == 1 and
                self.model.train_batch_size() == 64,
                'Formal Insert requires ZeRO1/global batch64')
        require(str(self.model.optimizer.gradient_accumulation_dtype) == 'torch.bfloat16',
                'Formal Insert requires BF16 gradient accumulation')
        write_new(ROOT / f'formal-start-rank-{self.rank}.json', {
            'utc': now(), 'rank': self.rank, 'pid': os.getpid(), 'start_step': 0,
            'engine_step': int(self.model.global_steps), 'resume': None,
            'config_sha256': digest(CONFIG), 'authorization_sha256': digest(AUTHORIZATION),
            'scheduler_total_steps': int(self.accelerator.deepspeed_engine_wrapped.engine.lr_scheduler.total_num_steps),
            'world_size': 4, 'batch_size': 4, 'gradient_accumulation_steps': 4,
            'runtime_env': {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE}})
        self.update_log = ROOT / f'updates-formal-rank-{self.rank}.jsonl'
        with self.update_log.open('x'):
            pass

    def _deepspeed_grad_norm(self, loss):
        value = super()._deepspeed_grad_norm(loss)
        self.latest_grad_norm = value.detach()
        return value

    def _accumulate_train_interval(self, loss, loss_dict, *args):
        finite = torch.stack([torch.isfinite(loss.detach()),
                              *[torch.isfinite(value).all() for value in loss_dict.values()]]).all()
        self.loss_health.append(finite)
        self.micro_in_update += 1
        boundary = self.accelerator.sync_gradients
        require(boundary == (self.micro_in_update == self.gradient_accumulation_steps),
                'Formal Insert gradient accumulation boundary changed')
        if boundary:
            require(bool(torch.stack(self.loss_health).all()),
                    'Formal Insert loss became nonfinite')
            require(self.latest_grad_norm is not None,
                    'Formal Insert global gradient proof is missing')
            grad_norm = float(self.latest_grad_norm.detach().float().cpu())
            require(math.isfinite(grad_norm), 'Formal Insert global gradient became nonfinite')
            step = self.global_step + 1
            require(self.model.global_steps == step,
                    'Formal Insert DeepSpeed update is repeated or skipped')
            total = torch.cuda.mem_get_info()[1]
            peak = torch.cuda.max_memory_allocated()
            measurement = {'step': step, 'engine_step': int(self.model.global_steps),
                           'pid': os.getpid(), 'loss_finite': True,
                           'loss': float(loss.detach().float().cpu()),
                           'global_grad_norm': grad_norm,
                           'microbatches': self.micro_in_update}
            measurement.update(check_headroom(total, peak, f'formal measurement step {step}'))
            self.measurements.append(measurement)
            with self.update_log.open('a') as stream:
                stream.write(json.dumps({'utc': now(), 'rank': self.rank, **measurement}, allow_nan=False) + '\n')
            self.loss_health.clear()
            self.micro_in_update = 0
        super()._accumulate_train_interval(loss, loss_dict, *args)

    def save_checkpoint(self):
        distributed_save_gate(self.accelerator, 'formal', self.global_step)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = super().save_checkpoint()
        torch.cuda.synchronize()
        total = torch.cuda.mem_get_info()[1]
        peak = torch.cuda.max_memory_allocated()
        self.save_peaks.append({'step': self.global_step, 'pid': os.getpid(),
                                **check_headroom(total, peak,
                                                 f'formal save step {self.global_step}')})
        self.accelerator.wait_for_everyone()
        return result

    def evaluate(self):
        torch.cuda.reset_peak_memory_stats()
        result = super().evaluate()
        if result is not None:
            require(all(math.isfinite(value) for value in result.values()
                        if isinstance(value, (int, float))),
                    'Formal Insert evaluation became nonfinite')
            torch.cuda.synchronize()
            total = torch.cuda.mem_get_info()[1]
            peak = torch.cuda.max_memory_allocated()
            self.evaluation_peaks.append({
                'step': self.global_step, 'pid': os.getpid(),
                **check_headroom(total, peak,
                                 f'formal evaluation step {self.global_step}')})
        return result

    def train(self):
        super().train()
        require(self.global_step == self.model.global_steps == self.max_steps == 20000,
                'Formal Insert did not reach the absolute final step')
        require([item['step'] for item in self.measurements] == list(range(1, 20001)),
                'Formal Insert update evidence is incomplete or out of order')
        expected_saves = list(range(self.save_every, self.max_steps + 1, self.save_every))
        require([item['step'] for item in self.save_peaks] == expected_saves,
                'Formal Insert save evidence is incomplete or out of order')
        scheduler = self.accelerator.deepspeed_engine_wrapped.engine.lr_scheduler
        runtime_record = {
            'stage': 'formal', 'rank': self.rank, 'pid': os.getpid(),
            'config_sha256': digest(CONFIG), 'authorization_sha256': digest(AUTHORIZATION),
            'output_dir': str(FORMAL_OUTPUT),
            'start_step': self.start_step, 'final_step': self.global_step,
            'trainer_initial_step': 0, 'engine_initial_step': 0,
            'engine_step': int(self.model.global_steps), 'trainer_step': self.global_step,
            'world_size': self.accelerator.num_processes,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'global_batch': self.model.train_batch_size(),
            'zero_stage': self.model.zero_optimization_stage(),
            'grad_accum_dtype': str(self.model.optimizer.gradient_accumulation_dtype),
            'runtime_env': {'NCCL_CUMEM_HOST_ENABLE': NCCL_CUMEM_HOST_ENABLE},
            'scheduler_total_steps': int(scheduler.total_num_steps),
            'measurements': self.measurements,
            'save_peaks': self.save_peaks,
            'evaluation_expected': self.val_dataset is not None and self.eval_every > 0,
            'evaluation_peaks': self.evaluation_peaks,
        }
        write_new(FORMAL_OUTPUT / f'formal-runtime-rank-{self.rank}.json', runtime_record)
        self.accelerator.wait_for_everyone()


cfg = OmegaConf.load(CONFIG)
validate_config(read(ROOT / 'reference_baseline.json'), OmegaConf.to_container(cfg, resolve=True))
require(cfg.resume is None and cfg.max_steps == 20000,
        'Formal Insert must initialize WanRobot fresh')
trainer.EasyWAMTrainer = FormalTrainer
runtime.run_training(cfg)

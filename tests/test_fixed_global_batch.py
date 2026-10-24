"""Full accumulation windows must survive the one-episode dataset boundary."""
import pytest
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader


@pytest.mark.parametrize('accumulation,combined_micro', [(2, 32), (4, 16), (8, 8), (16, 4)])
def test_full_global_batch_across_short_epochs(accumulation, combined_micro):
    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin
    accelerator = Accelerator(cpu=True, gradient_accumulation_plugin=GradientAccumulationPlugin(
        num_steps=accumulation, sync_with_dataloader=False))
    loader = accelerator.prepare(DataLoader(torch.arange(455), batch_size=combined_micro))
    micro_steps = 0
    windows = []
    for _ in range(3):
        for _batch in loader:
            with accelerator.accumulate():
                micro_steps += 1
                if accelerator.sync_gradients:
                    windows.append(micro_steps)
                    micro_steps = 0
    assert len(windows) >= 20
    assert all(window == accumulation for window in windows)
    accelerator.end_training()


def test_entrypoint_uses_opt_in_accumulation_plugin(monkeypatch):
    import accelerate
    import runtime
    captured = {}

    class StopAtConstructor(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise StopAtConstructor

    monkeypatch.setattr(accelerate, 'Accelerator', capture)
    cfg = OmegaConf.create({'mixed_precision': 'bf16', 'gradient_accumulation_steps': 8,
                           'gradient_accumulation_sync_with_dataloader': False})
    with pytest.raises(StopAtConstructor):
        runtime.run_training(cfg)
    assert 'gradient_accumulation_steps' not in captured
    plugin = captured['gradient_accumulation_plugin']
    assert plugin.num_steps == 8 and plugin.sync_with_dataloader is False

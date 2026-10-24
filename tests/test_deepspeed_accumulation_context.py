"""Exercise accumulation boundaries without loading a model or a GPU."""
from contextlib import contextmanager

import pytest
from trainer import _training_accumulation_context


@pytest.mark.parametrize("is_deepspeed", [False, True])
@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_accumulation_keeps_boundaries_without_deepspeed_no_sync(is_deepspeed, accumulation):
    class Accelerator:
        micro_steps = 0
        inside_no_sync = False

        @contextmanager
        def accumulate(self, *models):
            self.micro_steps += 1
            self.sync_gradients = self.micro_steps % accumulation == 0
            assert models == (() if is_deepspeed else (model,))
            self.inside_no_sync = bool(models) and not self.sync_gradients
            try:
                yield
            finally:
                self.inside_no_sync = False

    model = object()
    accelerator = Accelerator()
    updates = 0
    for _ in range(accumulation * 3):
        with _training_accumulation_context(accelerator, model, is_deepspeed=is_deepspeed):
            if is_deepspeed:
                # Accelerate's DeepSpeed wrapper invokes engine.step every microbatch.
                assert not accelerator.inside_no_sync
            if accelerator.sync_gradients:
                updates += 1
    assert accelerator.micro_steps == accumulation * 3
    assert updates == 3

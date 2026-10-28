import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler
from utils.samplers import ResumableEpochSampler as Sampler


def shard(sampler, batch, world, rank):
    batches = BatchSampler(sampler, batch_size=batch, drop_last=False)
    if world > 1:
        batches = BatchSamplerShard(batches, num_processes=world, process_index=rank,
                                   split_batches=False, even_batches=True)
    return list(batches)


@pytest.mark.parametrize("size", [1, 7, 15, 16, 17, 455])
@pytest.mark.parametrize("batch", [1, 4, 8, 16])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_absolute_epoch_and_every_resume_offset(size, batch, world):
    for epoch in (0, 3):
        for rank in range(world):
            permutation = torch.randperm(size, generator=torch.Generator().manual_seed(42 + epoch)).tolist()
            expected = shard(permutation, batch, world, rank)
            for offset in range(len(expected)):
                sampler = Sampler(range(size), 42, batch, world)
                sampler.set_epoch_offset(91)
                sampler.set_absolute_epoch(epoch)
                sampler.set_epoch(127)  # Accelerate's relative counter must not replace it.
                sampler.set_resume_batch_offset(offset)
                assert shard(sampler, batch, world, rank) == expected[offset:]
                remaining = ((len(expected) - offset) * batch * world
                             if world > 1 else size - offset * batch)
                assert len(sampler) == remaining
            sampler.set_resume_batch_offset(len(expected))
            assert len(sampler) == 0


def test_epochs_change_and_negative_resume_is_rejected():
    sampler = Sampler(range(455), 42, 8, 2)
    sampler.set_absolute_epoch(0)
    first = list(sampler)
    sampler.set_absolute_epoch(1)
    assert list(sampler) != first
    with pytest.raises(ValueError):
        sampler.set_resume_batch_offset(-1)
    sampler.set_resume_batch_offset(30)
    with pytest.raises(ValueError):
        list(sampler)


def test_legacy_sampler_api_is_preserved():
    sampler = Sampler(range(455), 42, 8, 2)
    sampler.set_epoch_offset(3)
    sampler.set_resume_batch_offset(2)
    expected = torch.randperm(455, generator=torch.Generator().manual_seed(45)).tolist()
    assert list(sampler) == expected[32:]
    assert len(sampler) == 455
    sampler.set_epoch(1)
    assert list(sampler) == torch.randperm(455, generator=torch.Generator().manual_seed(46)).tolist()

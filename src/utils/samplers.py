from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self.absolute_epoch = None

    def set_absolute_epoch(self, epoch: int):
        self.absolute_epoch = int(epoch)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        if int(batch_in_epoch) < 0:
            raise ValueError("Resume batch offset must be nonnegative.")
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        epoch = self.epoch + self.epoch_offset if self.absolute_epoch is None else self.absolute_epoch
        g.manual_seed(self.seed + epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.absolute_epoch is not None:
            # Match Accelerate's full-epoch padding before discarding consumed batches.
            total = self._absolute_epoch_size()
            if indices and total > len(indices):
                original = list(indices)
                indices.extend(original[i % len(original)] for i in range(total - len(indices)))
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            if sample_offset > total:
                raise ValueError("Resume batch offset exceeds the padded epoch length.")
            return iter(indices[sample_offset:])
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        if self.absolute_epoch is not None:
            return max(0, self._absolute_epoch_size() - self.resume_batch_offset * self.batch_size * self.num_processes)
        return len(self.dataset)

    def _absolute_epoch_size(self) -> int:
        size = len(self.dataset)
        if self.num_processes > 1:
            chunk = self.batch_size * self.num_processes
            return ((size + chunk - 1) // chunk) * chunk
        return size

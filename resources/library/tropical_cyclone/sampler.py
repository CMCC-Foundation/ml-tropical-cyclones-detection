from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Sampler
import torch.distributed as dist
import torch
import math


class DistributedWeightedSampler(Sampler):
    def __init__(
        self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=False
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.total_num_samples = dataset.total_num_samples
        self.weights = [i / sum(self.total_num_samples) for i in self.total_num_samples]
        self.num_samples = [
            int(math.ceil(i * 1.0 / self.num_replicas)) for i in self.total_num_samples
        ]
        self.total_size = [i * self.num_replicas for i in self.num_samples]
        self.replacement = replacement
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, num_samples in enumerate(self.total_num_samples):
            if self.shuffle:
                indices.append(torch.randperm(num_samples, generator=g).tolist())
            else:
                indices.append(list(range(num_samples)))
            indices[i] += indices[i][: (self.total_size[i] - len(indices[i]))]
            assert len(indices[i]) == self.total_size[i]
            indices[i] = indices[i][self.rank : self.total_size[i] : self.num_replicas]
            assert len(indices[i]) == self.num_samples[i]
        # create the weighted random sampler
        sampler = list(
            WeightedRandomSampler(
                weights=self.weights, num_samples=sum(self.num_samples)
            )
        )
        # create indices array
        data_indices = []
        ijk = [0] * len(indices)
        for bucket in sampler:
            data_indices.append(indices[bucket][ijk[bucket] % len(indices[bucket])])
            ijk[bucket] += 1
        return iter(zip(sampler, data_indices))

    def __len__(self):
        return sum(self.num_samples)

    def set_epoch(self, epoch):
        self.epoch = epoch

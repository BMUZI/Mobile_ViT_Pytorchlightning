import torch
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
import math, random
from tqdm import tqdm

class MultiScaleSampler(data.Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle

        self.num_samples = len(self.dataset)
        self.indices = list(range(self.num_samples))

        self.weights = []

        for i, (img, _) in enumerate(tqdm(self.dataset, desc='MultiScaleSampler')):
            weight = img.size()[1] * img.size()[2] # height * width
            self.weights.append(weight)
        self.weights = torch.tensor(self.weights, dtype=torch.double)
        self.weights /= self.weights.sum()


        # Determine number of samples for each replica
        self.num_samples_per_replica = int(math.ceil(self.num_samples * 1.0 / self.num_replicas)) if self.num_replicas is not None else self.num_samples

    def __iter__(self):
        # Shuffle indices
        if self.shuffle:
            self._shuffle_indices()

        # Divide indices among replicas
        if self.num_replicas is not None and self.rank is not None:
            self.indices = self.indices[self.rank:self.num_samples:self.num_replicas]

        # Batch indices
        batch_indices = [self.indices[i:i+self.batch_size] for i in range(0, len(self.indices), self.batch_size)]

        # Yield batches using weighted random sampler for each batch
        for indices in batch_indices:
            weights = self.weights[indices]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            yield from sampler

    def __len__(self):
        return self.num_samples_per_replica

    def _shuffle_indices(self):
        random.seed(self.epoch)
        random.shuffle(self.indices)
        self.epoch += 1


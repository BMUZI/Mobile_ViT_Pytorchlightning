import torch
from torch.utils.data.sampler import Sampler
import math
import random
import numpy as np

class MultiScaleSamplerDDP(Sampler):
    def __init__(self, base_im_w: int, base_im_h: int, base_batch_size: int, n_data_samples: int, min_scale_mult: float = 0.5, max_scale_mult: float = 1.5, n_scales: int = 5, is_training: bool = False) -> None:
    
        min_im_w, max_im_w = int(base_im_w * min_scale_mult), int(base_im_w * max_scale_mult)
        min_im_h, max_im_h = int(base_im_h * min_scale_mult), int(base_im_h * max_scale_mult)
    
        num_replicas = torch.distributed.get_world_size()
        rank =  torch.distributed.get_rank ()

        num_samples_per_replica = int (math.ceil(n_data_samples * 1.0 / num_replicas))
        total_size = num_samples_per_replica * num_replicas
        img_indices = [idx for idx in range(n_data_samples)]
        img_indices += img_indices[:(total_size - n_data_samples)]
        assert len(img_indices) == total_size

        self.shuffle = False

        if is_training:
            width_dims = list(np.linspace (min_im_w, max_im_w, n_scales))
            height_dims = list(np.linspace (min_im_h, max_im_h, n_scales))
            width_dims = [(w // 32) * 32 for w in width_dims]
            height_dims = [(h // 32) * 32 for h in height_dims]

            img_batch_pairs = list()
            base_elements = base_im_w  * base_im_h * base_batch_size

            for (h, w) in zip (height_dims, width_dims):
                batch_size = max(1, (base_elements / (h * w)))
                img_batch_pairs.append((h, w, batch_size))
            self.img_batch_pairs = img_batch_pairs
            self.shuffle = True
        else:
            self.img_batch_pairs = [(base_im_w  * base_im_h * base_batch_size)]

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.epoch = 0
        self.rank = rank
        self.num.replicas = num_replicas

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random. shuffle (self.img_indices)
            random. shuffle (self.img_batch_pairs)
            indices_rank_i= self.img_indices[self.rank:len(self.img_indices) : self.num_replicas]
        else:
            indices_rank_i= self.img_indices[self.rank:len(self.img_indices) : self.num_replicas]

        start_index = 0

        while start_index < self.n_samples_per_replica:
                curr_h, curr_w, curr_bsz = random.choice(self.img_batch_pairs)
                end_index = min(start_index + curr_bsz, self.n_samples_per_replica)
                batch_ids = indices_rank_i[start_index: end_index]
                n_batch_samples = len(batch_ids)

                if n_batch_samples != curr_bsz:
                    batch_ids += indices_rank_i[:(curr_bsz - n_batch_samples)]
                
                start_index += curr_bsz

                if len(batch_ids) > 0:
                    batch = [(curr_h, curr_w, b_id) for b_id in batch_ids]
                    yield batch
    
    def set_epoch (self, epoch: int) -> None:
        self.epoch = epoch
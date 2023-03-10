# Import basic pkg
from torch.utils.data import DataLoader

# Import custom pkg
from dataset.dataset import get_datasets
from dataset.multi_scale import MultiScaleSampler
from dataset.multi_scale_torchdistributed import MultiScaleSamplerDDP

# Get dataloaders
def get_dataloaders(args):

    # Datasets
    data_set_select = get_datasets(args)
    
    data_loader_select = {
        'train_aug': DataLoader(dataset = data_set_select['validation'],
                                batch_sampler = MultiScaleSampler(data_set_select['train_aug'], batch_size=120, shuffle=True),
                                num_workers = args.num_workers,
                                pin_memory = True),
        'validation': DataLoader(dataset = data_set_select['validation'],
                    shuffle=False,
                    batch_size = args.batch_size,
                    num_workers = args.num_workers,
                    pin_memory = True)

#        'train_aug': DataLoader(dataset = data_set_select['train_aug'],
#                            batch_size=None,
#                            pin_memory = True,
#                            num_workers = args.num_workers,
#                            batch_sampler = MultiScaleSamplerPL(256, 256, args.batch_size,
#                                                                        base_batch_size=args.batch_size,
#                                                                        n_data_samples = len(data_set_select['train_aug']),
#                                                                        batch_size = (args.batch_size * args.gpus),
#                                                                        drop_last = True,
#                                                                        is_training=True),),
#        'validation': DataLoader(dataset = data_set_select['validation'],
##                            batch_size=None,
#                            pin_memory = True,
#                            num_workers = args.num_workers,
#                            batch_sampler = MultiScaleSamplerPL(256, 256, args.batch_size,
#                                                                        base_batch_size=args.batch_size,
#                                                                        n_data_samples = len(data_set_select['train_aug']),
#                                                                        batch_size = (args.batch_size * args.gpus),
#                                                                        drop_last = True,
#                                                                        is_training=True),)
    }
    return data_loader_select
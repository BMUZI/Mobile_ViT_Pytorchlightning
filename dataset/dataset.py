# Import basic pkg
import os
from torchvision import transforms

from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageNet#, ImageFolder

# Import custom pkg
from dataset.transform import transform_cifar100_train, transform_cifar100_val, transform_imagenet_train, transform_imagenet_val

# Get datasets
def get_datasets(args):

    # Data root 
    data_root = os.path.join(args.data_root, args.data)

    # Data root 
    data_root = os.path.join(args.data_root, args.data)
    if args.data == 'imagenet' or args.data == 'ImageNet':
        data_root = os.path.join(data_root, 'Original')
    elif args.data == 'cifar100':
        data_root = os.path.join(data_root, 'pytorch')
    elif args.data == 'cifar10':
        data_root = os.path.join(data_root)
    elif args.data == 'mnist':
        data_root = os.path.join(data_root)
    
    if args.data == 'imagenet' or args.data == 'ImageNet':
        data_set_select = {
            'train_aug': ImageNet(transform = transform_imagenet_train,
                                split='train',
                                root = data_root),
            'validation': ImageNet(transform = transform_imagenet_val,
                                split='val',
                                root = data_root),
        }
    elif args.data == 'cifar10' or 'cifar100' or 'mnist':
        data_set_select = {
            'train_aug': CIFAR100(train = True,
                                transform = transform_cifar100_train,
                                root = data_root,
                                download = False),
            'validation': CIFAR100(train = False,
                                transform = transform_cifar100_val,
                                root = data_root,
                                download = False)
        }
    else:
        raise Exception(f'MODEL "{args.model}" not supported.')

    return data_set_select
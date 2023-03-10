# Import custom pkg
from models.basic_mvit_3dim import MobileViT
from models.basic_mvit import MobileViT_
from models.shortcut_mvit import ShortCutMobileViT
from models.shortcut_mvit_cifar import ShortCutMobileViT_CIFAR
from models.coatnet import CoAtNet

def get_model(args):
    model_str = args.architecture

    if model_str == 'mobilevit_xxs':
        return MVIT_(args)
    
    elif model_str == 'mobilevit_xs':
        return MVIT_(args)

    elif model_str == 'mobilevit_s':
        return MVIT_(args)

    elif model_str == 'cifarmvit_s':
        return MVIT_(args)
        
    elif model_str == 'coatnet_0':
        return coatnet(args)
    
    elif model_str == 'coatnet_1':
        return coatnet(args)

    elif model_str == 'coatnet_2':
        return coatnet(args)

    elif model_str == 'coatnet_3':
        return coatnet(args)

    elif model_str == 'coatnet_4':
        return coatnet(args)

    elif model_str == 'mvitc':
        return MVITC(args)
    
    elif model_str == 'mvitc_cifar_64' or 'mvitc_cifar_256':
        return MVITCCIFAR(args)
    
    elif model_str == 'omvit':
        return OMVIT(args)

    else:
        raise Exception(f'Model "{model_str}" not supported.')

def OMVIT(args):
    if args.architecture == 'omvit':
        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        return MobileViT((256, 256), dims, channels, num_classes=1000)

    else:
        raise Exception(f'Mode "{args.architecture}" not supported.')

def MVITC(args):
    if args.architecture == 'mvitc':
        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        return ShortCutMobileViT((256, 256), dims, channels, num_classes=1000)
    
    else:
        raise Exception(f'Mode "{args.architecture}" not supported.')
 
def MVITCCIFAR(args):
    if args.architecture == 'mvitc_cifar_64':
        dims = [144]
        channels = [16, 32, 64, 64]
        return ShortCutMobileViT_CIFAR((64, 64), dims, channels, num_classes=100)
    
    elif args.architecture == 'mvitc_cifar_256':
        dims = [256]
        channels = [16, 64, 256, 256]
        return ShortCutMobileViT_CIFAR((64, 64), dims, channels, num_classes=100)
    
    else:
        raise Exception(f'Mode "{args.architecture}" not supported.')

def MVIT_(args):
    if args.architecture == 'mobilevit_xxs':
        dims = [64, 80, 96]
        channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
        return MobileViT_((256, 256), dims, channels, num_classes=1000, expansion=2)

    elif args.architecture == 'mobilevit_xs':
        dims = [96, 120, 144]
        channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
        return MobileViT_((256, 256), dims, channels, num_classes=1000)


    elif args.architecture == 'mobilevit_s':
        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        return MobileViT_((256, 256), dims, channels, num_classes=1000)

    elif args.architecture == 'cifarmvit_s':
        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        return MobileViT_((64, 64), dims, channels, num_classes=100)

    else:
        raise Exception(f'Mode "{args.architecture}" not supported.')

def coatnet(args):
    if args.architecture == 'coatnet_0':
        num_blocks = [2, 2, 3, 5, 2]            # L
        channels = [64, 96, 192, 384, 768]      # D
        return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


    elif args.architecture == 'coatnet_1':
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [64, 96, 192, 384, 768]      # D
        return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


    elif args.architecture == 'coatnet_2':
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [128, 128, 256, 512, 1026]   # D
        return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


    elif args.architecture == 'coatnet_3':
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [192, 192, 384, 768, 1536]   # D
        return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


    elif args.architecture == 'coatnet_4':
        num_blocks = [2, 2, 12, 28, 2]          # L
        channels = [192, 192, 384, 768, 1536]   # D
        return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)
    
    else:
        raise Exception(f'Mode "{args.architecture}" not supported.')
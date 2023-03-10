# Import custom pkg
from models.basic_mvit_3dim import MobileViT
from models.basic_mvit import MobileViT_
from models.shortcut_mvit import ShortCutMobileViT
from models.shortcut_mvit_conv import ShortCutMobileViT_CONV
from models.shortcut_mvit_cifar import ShortCutMobileViT_CIFAR
from models.shortcut_mvit_cifar_7block import ShortCutMobileViT_CIFAR_7block

#from models.coatnet import CoAtNet

def get_model(args):
    model_args = making(args)

    if model_args['model_str'] in ['omvit']:
        return MobileViT(image_size = model_args['input_size'],
                         dims = model_args['dims'],
                         channels = model_args['channels'], 
                         num_classes = model_args['num_classes'], )
    
    elif model_args['model_str'] in ['mvitc_cifar_64', 'mvitc_cifar_256']:
        return ShortCutMobileViT_CIFAR(image_size = model_args['input_size'],
                         dims = model_args['dims'],
                         channels = model_args['channels'], 
                         num_classes = model_args['num_classes'], )
    
    elif model_args['model_str'] in ['mvitc_cifar_7block']:
        return ShortCutMobileViT_CIFAR_7block(image_size = model_args['input_size'],
                         dims = model_args['dims'],
                         channels = model_args['channels'], 
                         num_classes = model_args['num_classes'], )

    elif model_args['model_str'] in ['mvitc']:
        return ShortCutMobileViT(image_size = model_args['input_size'],
                         dims = model_args['dims'],
                         channels = model_args['channels'], 
                         num_classes = model_args['num_classes'], )
        
    elif model_args['model_str'] in ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'cifarmvit_s']:
        return MobileViT_(image_size = model_args['input_size'],
                         dims = model_args['dims'],
                         channels = model_args['channels'], 
                         num_classes = model_args['num_classes'], )
    
    elif model_args['model_str'] in ['mvitc_conv']:
        return MobileViT_(image_size = model_args['input_size'],
                    dims = model_args['dims'],
                    channels = model_args['channels'], 
                    num_classes = model_args['num_classes'], )

def making(args):
    model_str = args.architecture

    if model_str in ['mobilevit_xxs']:
        dims = [64, 80, 96]
        channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
        input_size = (256, 256)
        num_classes = 1000

    elif model_str in ['mobilevit_xs']:
        dims = [96, 120, 144]
        channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
        input_size = (256, 256)
        num_classes = 1000

    elif model_str in ['omvit', 'mvitc', 'mobilevit_s', 'cifarmvit_s', 'mvitc_conv']:
        dims = [144, 192, 240]
        channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        input_size = (256, 256)
        num_classes = 1000

    elif model_str in ['mvitc_cifar_64']:
        dims = [144]
        channels = [16, 32, 64, 64]
        input_size = (32, 32)
        num_classes = 100

    elif model_str in ['mvitc_cifar_256']:
        dims = [256]
        channels = [16, 64, 256, 256]
        input_size = (32, 32)
        num_classes = 100

    elif model_str in ['mvitc_cifar_7block']:
        dims = [144, 192]
        channels = [16, 32, 32, 64, 64, 128, 128, 256]
        input_size = (32, 32)
        num_classes = 100

    else:
        raise Exception(f'Model "{model_str}" not supported.')

    data_loader_select = {
        'model_str': model_str,
        'dims': dims,
        'channels': channels,
        'input_size': input_size,
        'num_classes': num_classes,
    }

    return data_loader_select
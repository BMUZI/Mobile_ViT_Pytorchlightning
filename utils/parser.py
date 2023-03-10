# Import basic pkg
import argparse
import json

def get_args(stdin):
    parser = argparse.ArgumentParser("MVIT_GDY")

    # System
    parser.add_argument('--run_label', type = str, default ='train', help ='run_label')
    parser.add_argument('--log_freq', type = int, default = 50, help ='log_freq')
    parser.add_argument('--devices', type = str,default ='gpu', help ='devices')
    parser.add_argument('--gpus', type = int, default = 2, help ='gpus')

    # Data Loader
    parser.add_argument('--data_root', type = str, default = '/home/hjkim/database/', help='data_root')
    parser.add_argument('--data', type = str, default = 'imagenet', help='data')
#    parser.add_argument('--data_download', type=str, action='store', default=False, help='data_download')
    
    parser.add_argument('--n_classes', type = int, default = 100, help='n_classes')
    parser.add_argument('--batch_size', type = int, default = 512, help='batch_size')
    parser.add_argument('--num_workers', type = int, default = 20, help='num_workers')

    # Transpose
    parser.add_argument('--shuffle', type=str, action='store', default = True, help='shuffle')
#    parser.add_argument('--drop_last', type=str, action='store', default=False, help='drop_last')

    # Logger
    parser.add_argument('--log_dir', type = str, default = '../logs', help='log_dir')
    parser.add_argument('--log_name', type = str, default = 'imagenet_mvitv1', help='log_name')
    parser.add_argument('--version', type = int, default = 0, help='version')
    parser.add_argument('--checkpoint', type=str, default = 'last.ckpt', help='checkpoint')

    # Loss
#    parser.add_argument('--loss_name', type=str, action='store', default='cross_entropy', help='loss_name')
#    parser.add_argument('--loss_label_smoothing', type=float, default=0.1, help='loss_label_smoothing')

    # Optimizer
#    parser.add_argument('--optimizer', type=str, action='store', default='AdamW', help='optimizer')
    parser.add_argument('--learning_rate', type = float, default = 1e-4, help ='learning_rate')
    parser.add_argument('--start_factor', type = float, default = 1e-1, help ='start_factor')
    parser.add_argument('--lr_max_time', type = float, default = 0.04, help ='lr_max_time')
    parser.add_argument('--optimizer_weight_decay', type = float, default = 1e-3, help ='optimizer_weight_decay')

    # Model
    parser.add_argument('--architecture', type = str, default ='cifarmvit_s', help ='architecture')

    # Trainer
    parser.add_argument('--epoch', type = int, default = 300, help='n_epochs')
    parser.add_argument('--mode', type = str,default = 'train', help='mode')
#    parser.add_argument('--ema', type=str, default = True, help='ema')

    # Metric
    parser.add_argument('--checkpoint_metric', type = str, default = 'val_acc_1', help ='checkpoint_metric')
    parser.add_argument('--checkpoint_mode', type = str, default = 'max', help ='checkpoint_metric')
    parser.add_argument('--saving_metric_num', type = int, default = 3, help ='saving_metric_num')

    #system
    args = parser.parse_args()

    args.__class__.__repr__ = lambda x: 'Input args: ' + json.dumps(x.__dict__, indent=4)
    return args

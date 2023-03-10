# Import basic pkg
import sys, os
from datetime import datetime
from torchsummary import summary as summary
import yaml, torch

# Add addtional pkg path
prjdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prjdir)

# Import custom pkg
from utils.parser import get_args
from utils.logger import get_logger
from utils.trainer import get_trainer
#from dataset.dataloader import get_dataloaders
from dataset.dataloader import get_dataloaders
#from models.model_utils import Classifier
from models.model_utils_weight import Classifier

# Main
def main(args):

    # Logger
    CSVLogger, TB_logger = get_logger(args)

    # Model
    model = Classifier(args, TB_logger)
    
    if args.mode == 'summary':
        summary(model, (3, 256, 256), device = 'cpu')
        raise Exception("Model Printed")
    
    elif args.mode == 'print':
        raise Exception("Model Printed")

    # Trainer
    trainer = get_trainer(args, CSVLogger, TB_logger)

    # Dataloaders
    data_loader = get_dataloaders(args)

    if input("Press 1 to Use GPU [0, 1, 2]? : ") == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
        
    if args.mode == 'train':
        trainer.fit(model, data_loader['train_aug'], data_loader['validation'])
        trainer.test(model = None, test_dataloaders = data_loader['validation'])

    elif args.mode == 'test':
        trainer.test(model = model, test_dataloaders = data_loader['validation'])

    else:
        raise Exception(f'Mode "{args.mode}" not supported.')

if __name__ == '__main__':

    # Args
    args = get_args(sys.stdin)

    main(args)

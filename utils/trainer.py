# Import basic pkg
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


# Trainer
def get_trainer(args, CSVLogger, TB_logger):
    if os.path.exists(os.path.join(args.log_dir, args.log_name, ('version_' + str(args.version)), args.checkpoint)):
        check_point = os.path.join(args.log_dir, args.log_name, ('version_' + str(args.version)), args.checkpoint)
    else:
        check_point = None

    if os.path.exists(os.path.join(args.log_dir, args.log_name, ('version_' + str(args.version)))):
        train_version = args.version + 1
    else:
        train_version = args.version

    if args.mode == 'train' or args.mode == 'test':
        checkpoint_callback = ModelCheckpoint(
                            monitor=args.checkpoint_metric,
                            dirpath=os.path.join(args.log_dir, args.log_name, ('version_' + str(train_version))),
                            filename='model-{epoch:02d}-{val_acc_1:.2f}',
                            save_top_k=3,
                            mode=args.checkpoint_mode,
                            save_last=True
                            )

        trainer = pl.Trainer(accelerator = args.devices,
                            devices = args.gpus,
                            max_epochs = args.epoch,
                            logger = [CSVLogger, TB_logger],
                            strategy = 'ddp_find_unused_parameters_false',
                            callbacks = checkpoint_callback,
                            default_root_dir = args.log_dir,
                            resume_from_checkpoint = check_point,
                            )

    else:
        raise Exception(f'Mode "{args.mode}" not supported.')

    return trainer
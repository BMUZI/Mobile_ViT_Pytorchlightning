# Import basic pkg
import os
from pytorch_lightning import loggers as pl_loggers

# Set Logger
def get_logger(args):
    if os.path.exists(os.path.join(args.log_dir, args.log_name, ('version_' + str(args.version)))):
        train_version = args.version + 1
    else:
        train_version = args.version

    CSVLogger = pl_loggers.CSVLogger(save_dir = args.log_dir,
                                    name = args.log_name,
                                    version = train_version,
                                    flush_logs_every_n_steps = args.log_freq)
    #CSVLogger.log_hyperparams(args)

    TB_logger = pl_loggers.TensorBoardLogger(save_dir = args.log_dir,
                                    name = args.log_name,
                                    version = train_version,
                                    default_hp_metric=False)

    return CSVLogger, TB_logger
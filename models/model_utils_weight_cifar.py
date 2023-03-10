# Import basic pkg
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim

# Import custom pkg
from models.making_model import get_model
from utils.smooth_croos_entropy import LabelSmoothingCrossEntropyLoss

class Classifier(pl.LightningModule):
    def __init__(self, args, TB_logger):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = get_model(args)
        self.TB_logger = TB_logger
        self.loss_fn = LabelSmoothingCrossEntropyLoss(smoothing=0.1)

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx, part='train'):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc_1 = (1.0 * (F.softmax(logits, 1).argmax(1) == y)).mean()
        acc_5 = (1.0 * (torch.topk(F.softmax(logits, 1), k=5, dim=1)[1] == y.view(-1, 1)).sum(dim=1).bool().float().mean())
        
        # logger.info(acc)
        self.log(f'{part}_loss', loss, prog_bar=True)
        self.log(f'{part}_acc_1', acc_1, prog_bar=True)
        self.log(f'{part}_acc_5', acc_5, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, part='val')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = self.args.learning_rate, weight_decay = self.args.optimizer_weight_decay)
        
        # Define the learning rate scheduler
        total_iterations = self.trainer.max_steps
        warmup_iterations = int(self.args.lr_max_time * self.trainer.max_steps)
        linear_scheduler = optim.lr_scheduler.LinearLR(optimizer, 
                                                    start_factor = self.args.start_factor, 
                                                    end_factor=1.0, 
                                                    total_iters=warmup_iterations)
        
        # Define the cosine annealing scheduler
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=total_iterations - warmup_iterations, 
                                                                eta_min = (self.args.learning_rate) * self.args.start_factor)
        
        return [optimizer], [linear_scheduler, cosine_scheduler]




    def training_epoch_end(self, outputs):
        # log histogram of parameters
        for name, param in self.named_parameters():
            self.TB_logger.experiment.add_histogram(name, param, self.current_epoch)
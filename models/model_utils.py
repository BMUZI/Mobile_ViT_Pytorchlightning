# Import basic pkg
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam

# Import custom pkg
from models.making_model import get_model

class Classifier(pl.LightningModule):
    def __init__(self, args, TB_logger):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = get_model(args)
        self.TB_logger = TB_logger

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx, part='train'):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (1.0 * (F.softmax(logits, 1).argmax(1) == y)).mean()
        # logger.info(acc)

        self.log(f'{part}_loss', loss, prog_bar=True)
        self.log(f'{part}_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, part='val')

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.args.learning_rate, weight_decay=self.args.optimizer_weight_decay)

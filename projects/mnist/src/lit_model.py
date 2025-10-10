import torch
import torch.nn as nn
import pytorch_lightning as pl

from projects.mnist.src.model import Model

class LitModel(pl.LightningModule):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = Model()
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.net(x)


    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/acc", acc, prog_bar=True)
        return loss


    def training_step(self, batch, _):
        return self._step(batch, "train")


    def validation_step(self, batch, _):
        self._step(batch, "val")


    def test_step(self, batch, _):
        self._step(batch, "test")


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

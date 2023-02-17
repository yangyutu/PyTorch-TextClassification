from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy


class TextClassifier(pl.LightningModule):
    def __init__(self, config: Dict, model: nn.modules):
        super().__init__()
        self.config = config
        num_classes = config["num_classes"]

        self.model = model
        # if hasattr(self.model, "device"):
        #     self.model.device = self.device
        #     print(f"set model device to {self.device}")
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.val_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.test_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config["lr"])
        return optimizer

    def training_step(self, batch, batch_idx):

        x, target = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, target)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self.train_accuracy.update(logits, target)
        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log(
            "train_acc_epoch", self.train_accuracy.compute(), prog_bar=True, logger=True
        )
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):

        x, target = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, target)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self.val_accuracy.update(logits, target)

        return loss

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log(
            "val_acc_epoch", self.val_accuracy.compute(), prog_bar=True, logger=True
        )
        self.val_accuracy.reset()

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
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(num_classes=num_classes)
        self.val_accuracy = Accuracy(num_classes=num_classes)
        self.test_accuracy = Accuracy(num_classes=num_classes)

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


if __name__ == "__main__":
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
    config = {}

    config["num_classes"] = 3
    config["num_characters"] = len(alphabet) + 1
    config["max_seq_length"] = 1014
    config["dropout_input"] = 0.1
    config["most_frequent_words"] = 50000

    model = CharacterLevelCNN(config)

    print(model)

    model2 = LogisticBOWModel(config)

    print(model2)

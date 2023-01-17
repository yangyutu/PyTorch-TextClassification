from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# reference
# https://github.com/bentrevett/pytorch-sentiment-analysis


class LogisticBOWModel(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.vocab_size = vocab_size
        num_classes = num_classes

        self.linear = nn.Linear(self.vocab_size, num_classes)

    def forward(self, x):

        x = self.linear(x)

        return x

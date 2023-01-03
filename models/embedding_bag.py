import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingBagModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class):
        super().__init__()

        self.embedding_bag = nn.EmbeddingBag(
            vocab_size + 1, embedding_dim, padding_idx=vocab_size
        )

        self.fc = nn.Linear(embedding_dim, num_class)

        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding_bag.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x):
        # ids has size of [batch size, seq len]

        ids = x

        # embeddings has shape [batch size, emb dim]
        embeddings = self.embedding_bag(ids)

        return self.fc(embeddings)

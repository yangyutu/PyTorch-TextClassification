import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import pytorch_lightning as pl
from torchmetrics import Accuracy


# reference
# https://github.com/bentrevett/pytorch-sentiment-analysis


class LogisticBOWModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_features = self.config["most_frequent_words"]
        num_classes = config["num_classes"]

        self.linear = nn.Linear(self.num_features, num_classes)

    def forward(self, x):

        x = self.linear(x)

        return x


class BagEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # ids has size of [batch size, seq len]

        ids, length = x

        embeddings = self.embedding(ids)

        # embeddings = [batch size, seq_len,  emb dim]

        # average over seq_len axis
        pooled = F.avg_pool2d(embeddings, (embeddings.shape[1], 1)).squeeze(1)

        # pooled = [batch size, embedding_dim]

        return self.fc(pooled)


class TokenLevelCNN2D(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv_0 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(filter_sizes[0], embedding_dim),
        )

        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(filter_sizes[1], embedding_dim),
        )

        self.conv_2 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(filter_sizes[2], embedding_dim),
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ids has size of [batch size, seq len]

        ids, length = x

        embeddings = self.embedding(ids)

        # embeddings = [batch size, sent len, emb dim]

        embeddings = embeddings.unsqueeze(1)

        # embeddings = [batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embeddings).squeeze(3))
        conved_1 = F.relu(self.conv_1(embeddings).squeeze(3))
        conved_2 = F.relu(self.conv_2(embeddings).squeeze(3))

        # self.conv_0(embeddings) has shape [batch size, n_filters, seq_len - filter_sizes[0] + 1, 1]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # pooling across the last axis

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


class TokenLevelCNN1D(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,  # each embedding dim is a feature channel
                    out_channels=n_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ids has size of [batch size, seq len]

        ids, length = x

        embeddings = self.embedding(ids)

        # embeddings = [batch size, sent len, emb dim]

        embeddings = embeddings.permute(0, 2, 1)

        conved = [F.relu(conv(embeddings)) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooling across the last axis

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


class TokenLevelLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ids = [batch size, seq_len]
        # length = [batch size]
        ids, length = x
        embeddings = self.embedding(ids)
        embeddings = self.dropout(embeddings)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embeddings, length.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]

        # output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)

        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]
        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        return prediction


class CharacterLevelCNN(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        num_classes = config["num_classes"]
        num_characters = config["num_characters"]
        max_seq_length = config["max_seq_length"]

        self.dropout_input = nn.Dropout2d(config["dropout_input"])

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_characters, out_channels=256, kernel_size=7, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=0),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=0),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=0),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        input_shape = (
            128,  # batch_size
            num_characters,
            max_seq_length,
        )
        self.output_dimension = self._get_conv_output(input_shape)

        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dimension, 1024), nn.ReLU(), nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5))

        self.fc3 = nn.Linear(1024, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(num_classes=num_classes)
        self.val_accuracy = Accuracy(num_classes=num_classes)
        self.test_accuracy = Accuracy(num_classes=num_classes)

    def _initialize_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    def forward(self, x):
        x = self.dropout_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class GenericModel(pl.LightningModule):
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

    def test_step(self, batch, batch_idx):

        x, target = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, target)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self.test_accuracy.update(logits, target)

        return loss

    def test_epoch_end(self, outs):
        # log epoch metric
        self.log(
            "test_acc_epoch", self.test_accuracy.compute(), prog_bar=True, logger=True
        )
        self.test_accuracy.reset()


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

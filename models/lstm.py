import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLevelLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_class: int,
        n_layers: int = 2,
        bidirectional: bool = False,
        dropout: int = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size + 1, embedding_dim, padding_idx=vocab_size
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_class)
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

        # if we want to get the hidden state for every step, use the following to unpack.
        # output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)

        if self.lstm.bidirectional:
            # get the last layer's two-directional hidden representation as the final representation
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            # get the last layer's uni-directional hidden representation as the final representation
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]
        logits = self.fc(hidden)
        # prediction = [batch size, output dim]
        return logits


def _test_lstm():

    from data_utils.text_data import TokenizedTextDataModule

    data_module = TokenizedTextDataModule(
        "AG_NEWS",
        return_length=True,
    )

    train_dataloader = data_module.train_dataloader()

    model_lstm = TokenLevelLSTM(
        vocab_size=data_module.vocab_size,
        embedding_dim=100,
        num_class=data_module.label_size,
        hidden_dim=256,
    )

    for batch in train_dataloader:
        x, target = batch
        model_lstm(x)
        break


if __name__ == "__main__":
    _test_lstm()

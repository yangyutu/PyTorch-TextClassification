import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.bert.modeling_bert import BertEncoder

import math


class PositionalEncoding(nn.Module):
    # Absolute positional encoding, following the attention is all you need paper, introduces a notion of word order.
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[:, : token_embedding.size(1), :]
        )


class LearnablePositionEncoding(nn.Module):
    # The original BERT paper states that unlike transformers, positional and segment embeddings are learned.
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(LearnablePositionEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(maxlen, emb_size)
        self.register_buffer("position_ids", torch.arange(maxlen).expand((1, -1)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: Tensor):
        position_ids = self.position_ids[:, : token_embedding.size(1)]
        return self.dropout(token_embedding + self.pos_embedding(position_ids))


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        embed_size,
        vocab_size,
        dropout=0.1,
        position_encoding_type: str = "learnable",
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        assert position_encoding_type in ["learnable", "cosine"]
        if position_encoding_type == "learnable":
            self.pos_encoding = LearnablePositionEncoding(embed_size, dropout=dropout)
        elif position_encoding_type == "cosine":
            self.pos_encoding = PositionalEncoding(embed_size, dropout=dropout)

        self.embed_layer_norm = nn.LayerNorm(embed_size)

    def forward(self, input_ids):
        embedding = self.embed_layer_norm(
            self.pos_encoding(self.token_embed(input_ids))
        )
        return embedding


class CustomTransformerEncoder(nn.Module):
    def __init__(
        self,
        tokenizer_config_name: str,
        num_classes: int,
        num_layers: int = 2,
        truncate: int = 512,
        embed_size: int = 128,
        nhead: int = 2,
        dim_feedforward: int = 512,
        dim_model: int = 128,
        dropout: float = 0.1,
        position_encoding_type: str = "learnable",
        pooled_output_embedding: bool = False,
    ):
        super(CustomTransformerEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config_name, use_fast=True
        )
        self.truncate = truncate
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.pooled_output_embedding = pooled_output_embedding
        self.embedding_layer = EmbeddingLayer(
            embed_size=embed_size,
            vocab_size=self.tokenizer.vocab_size,
            dropout=dropout,
            position_encoding_type=position_encoding_type,
        )
        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.update(
            {
                "num_attention_heads": nhead,
                "hidden_size": dim_model,
                "intermediate_size": dim_feedforward,
                "num_hidden_layers": num_layers,
            }
        )
        self.encoder = BertEncoder(config)
        self.linear = nn.Linear(dim_model, num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.apply(self._init_weight_fn)

    def _init_weight_fn(self, module):
        classname = module.__class__.__name__
        if "Linear" == classname:
            # m.weight.data shoud be taken from a normal distribution
            module.weight.data.normal_(0.0, 0.02)
            # m.bias.data should be 0
            module.bias.data.fill_(0)
        elif "Embedding" == classname:
            # m.weight.data shoud be taken from a normal distribution
            module.weight.data.normal_(0.0, 0.02)

    def _mean_pooling(self, last_hidden_states, attention_mask):
        token_embeddings = last_hidden_states
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        pooled_embedding = torch.sum(
            token_embeddings * input_mask_expanded, axis=1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return pooled_embedding

    def forward(self, text_list):
        encoded_inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            max_length=self.truncate,
            truncation="longest_first",
            padding="max_length",
            add_special_tokens=False if self.pooled_output_embedding else True,
        )
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        embedding = self.embedding_layer(encoded_inputs["input_ids"])
        # convert original attention mask to additive attenion score mask
        atten_mask = encoded_inputs["attention_mask"]
        mask = (
            atten_mask.float()
            .masked_fill(atten_mask == 0, float("-inf"))
            .masked_fill(atten_mask == 1, float(0.0))
        )
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L994
        mask = mask[:, None, None, :]
        embedding = self.encoder(embedding, attention_mask=mask)
        encoded_embeddings = embedding.last_hidden_state
        if self.pooled_output_embedding:
            representation = self._mean_pooling(
                encoded_embeddings, encoded_inputs["attention_mask"]
            )
        else:
            representation = encoded_embeddings[:, 0, :]
        predictions = self.linear(representation)

        return predictions


class HFTransformerEncoder(nn.Module):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    # the explanation of attention mask in huggingface transformer https://huggingface.co/docs/transformers/glossary#attention-mask
    def __init__(
        self,
        model_config_name,
        num_layers=2,
        nhead: int = 2,
        dim_model: int = 128,
        dim_feedforward: int = 512,
        truncate: int = 512,
        num_classes=2,
    ):
        super().__init__()
        # just load model architecture without pretrained model weights
        # https://huggingface.co/docs/transformers/model_doc/auto?highlight=from_pretrained#transformers.AutoConfig.from_pretrained
        self.model_config = AutoConfig.from_pretrained(model_config_name)
        self.model_config.update(
            {
                "hidden_size": dim_model,
                "intermediate_size": dim_feedforward,
                "num_attention_heads": nhead,
                "num_hidden_layers": num_layers,
            }
        )
        self.encoder = AutoModel.from_config(self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config_name, use_fast=True)
        self.truncate = truncate
        self.linear = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, text_list):
        encoded_inputs = self.tokenizer(
            text_list,
            return_tensors="pt",
            max_length=self.truncate,
            truncation="longest_first",
            padding="max_length",
            add_special_tokens=True,
        )
        # encoded_inputs is a dictionary with three keys: input_ids, attention_mask, and token_type_ids
        # the position ids [0, 1, ..., seq_len - 1] will be generated by default on the fly
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        encoder_outputs = self.encoder(**encoded_inputs)
        # encoder_outputs have two elements: last_hidden_state (shape: batch_size x seq_len x hidden_dim) and pooler_output (shape: batch_size x hidden_dim)
        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
        # bert_output is the last layer's hidden state
        last_hidden_states = encoder_outputs.last_hidden_state
        cls_representations = last_hidden_states[:, 0, :]
        predictions = self.linear(cls_representations)

        return predictions


def _test_custom_transformer():
    model = CustomTransformerEncoder("bert-base-uncased", 2, 2)
    model.to("cuda")
    print(model)

    batch = ["hellow, world", "great", "hte oqiej elkwoi soierle"]
    out = model(batch)
    print(out)


def _test_huggingface_transformer():
    model = HFTransformerEncoder("bert-base-uncased", truncate=512)
    # print(model.model_config)
    # print(model)
    model.to("cuda")
    batch = ["hellow, world", "great", "hte oqiej elkwoi soierle"]
    out = model(batch)
    print(out)


if __name__ == "__main__":
    _test_custom_transformer()
    # _test_huggingface_transformer()

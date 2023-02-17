import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch import Tensor
import math
from models.utils import batch_to_device


class PretrainedBertEncoder(nn.Module):
    def __init__(self, pretrained_model_name, truncate, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, use_fast=True
        )
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


class PretrainedBertEncoderV2(nn.Module):
    def __init__(self, pretrained_model_name, truncate, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, use_fast=True
        )
        self.truncate = truncate
        self.linear = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, tokenized_inputs):
        # encoded_inputs is a dictionary with three keys: input_ids, attention_mask, and token_type_ids
        # the position ids [0, 1, ..., seq_len - 1] will be generated by default on the fly
        # encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        tokenized_inputs = batch_to_device(tokenized_inputs, target_device=self.device)
        encoder_outputs = self.encoder(**tokenized_inputs)
        # encoder_outputs have two elements: last_hidden_state (shape: batch_size x seq_len x hidden_dim) and pooler_output (shape: batch_size x hidden_dim)
        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel
        # bert_output is the last layer's hidden state
        last_hidden_states = encoder_outputs.last_hidden_state
        cls_representations = last_hidden_states[:, 0, :]
        predictions = self.linear(cls_representations)

        return predictions

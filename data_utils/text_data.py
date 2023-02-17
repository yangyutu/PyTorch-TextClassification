import importlib
from collections import Counter
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torchtext
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchtext import datasets
from torchtext.vocab import vocab as vocab_builder
from tqdm import tqdm
from transformers import AutoTokenizer


class TextIterData(IterableDataset):
    def __init__(self, text_data_name, partition="train"):
        super().__init__()
        self.data_iter = getattr(
            importlib.import_module("torchtext.datasets"), text_data_name
        )(split=partition)

    def __iter__(self):
        for label, line in self.data_iter:
            yield line, label


class TextData(Dataset):
    def __init__(self, text_data_name, partition="train"):
        super().__init__()
        self.data_iter = getattr(
            importlib.import_module("torchtext.datasets"), text_data_name
        )(split=partition)
        self.data = list(self.data_iter)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, line = self.data[index]
        return line, label


def collate_batch(batch, text_pipeline, label_pipeline):
    text_list, label_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return (text_list, offsets), label_list


def collate_and_pad_batch(
    batch, text_pipeline, label_pipeline, padding_value, return_length=False
):
    text_list, label_list = [], []
    length_list = []
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        if return_length:
            length_list.append(len(processed_text))

    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    text_tensor = pad_sequence(text_list, batch_first=True, padding_value=padding_value)
    if return_length:
        return (text_tensor, torch.tensor(length_list)), label_tensor
    else:
        return text_tensor, label_tensor


def text_collate_fn(data, label_pipeline):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    texts, labels = zip(*data)
    label_list = [label_pipeline(label) for label in labels]
    return list(texts), torch.tensor(label_list).long()


def ngram_collate_fn(data, text_pipeline, label_pipeline):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    texts, labels = zip(*data)
    label_list = [label_pipeline(label) for label in labels]
    text_features = np.array([text_pipeline(text) for text in texts])
    return torch.tensor(text_features).float(), torch.tensor(label_list).long()


class TokenizedTextDataModule(pl.LightningDataModule):
    """This data class aims to produces training and validation dataloader via function
    train_dataloader() and val_dataloader().

    dataloader is an iterator that produce a tuple of (2D tensor, 1D tensor label list).

    2D tensor has integer data type and the shape of (batch_size, seq_len). Integer value is obtained via a dictionary, which maps a token to an integer.
    seq_len is determined by the longest sequence in the batch. Padding value is equal to vocab_size

    """

    def __init__(
        self,
        text_data_name: str,
        vocab_min_freq: int = 1,
        batch_size: int = 64,
        num_workers: int = 8,
        return_length: bool = False,
    ):
        """_summary_

        Args:
            text_data_name (str): the name of the dataset from text claasification dataset in https://pytorch.org/text/stable/datasets.html
            vocab_min_freq (int, optional): minimum token frequency to count the token as a valid entry in the dictionary. Defaults to 1.
            batch_size (int, optional): batch size for dataloader. Defaults to 64.
            num_workers (int, optional): number of worker for dataloader. Defaults to 8.
        """
        super().__init__()

        self.vocab_min_freq = vocab_min_freq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_length = return_length
        self.setup(text_data_name)

    def setup(self, text_data_name: str):
        """
        Load text data (training and ) and build vocabulary
        Args:
            text_data_name (str): the name of the dataset from text claasification dataset in https://pytorch.org/text/stable/datasets.html
        """

        self.train_data = TextData(text_data_name, "train")
        self.val_data = TextData(text_data_name, "test")

        self.tokenizer = torchtext.data.utils.get_tokenizer(
            "basic_english", language="en"
        )
        self.build_vocab_and_label()

        self.text_pipeline = lambda x: [
            self.vocab[token] for token in self.tokenizer(x)
        ]
        self.label_pipeline = lambda x: self.label_map[x]

    def build_vocab_and_label(self):

        self.label_set = set()
        counter = Counter()
        print("building vocabulary and label")
        for (line, label) in tqdm(self.train_data):
            counter.update(self.tokenizer(line))
            self.label_set.add(label)
        self.vocab = vocab_builder(counter, min_freq=self.vocab_min_freq)
        self.label_map = dict(zip(self.label_set, range(len(self.label_set))))
        self.padding_value = self.vocab_size

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def label_size(self):
        return len(self.label_set)

    def get_collate(self):

        return partial(
            collate_and_pad_batch,
            text_pipeline=self.text_pipeline,
            label_pipeline=self.label_pipeline,
            padding_value=self.padding_value,
            return_length=self.return_length,
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.get_collate(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.get_collate(),
        )


class TextDataModule(pl.LightningDataModule):
    """This data class aims to produce training and validation dataloader via function
    train_dataloader() and val_dataloader().

    dataloader is an iterator that produces a tuple of (a list of raw text, 1D tensor label list).

    """

    def __init__(
        self,
        text_data_name,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup(text_data_name)

    def setup(self, text_data_name):

        self.train_data = TextData(text_data_name, "train")
        self.val_data = TextData(text_data_name, "test")
        self.build_vocab_and_label()
        self.label_pipeline = lambda x: self.label_map[x]

    def build_vocab_and_label(self):

        self.label_set = set()

        for (line, label) in self.train_data:
            self.label_set.add(label)
        self.label_map = dict(zip(self.label_set, range(len(self.label_set))))

    @property
    def label_size(self):
        return len(self.label_set)

    def get_collate(self):

        return partial(
            text_collate_fn,
            label_pipeline=self.label_pipeline,
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.get_collate(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.get_collate(),
        )


def text_collate_with_tokenization(data, tokenizer, label_pipeline, truncate):

    texts, labels = zip(*data)
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        max_length=truncate,
        truncation="longest_first",
        padding="max_length",
    )
    label_list = [label_pipeline(label) for label in labels]
    return encoded, torch.tensor(label_list).long()


class PretrainedTokenizedTextDataModule(pl.LightningDataModule):
    """This data class aims to produce training and validation dataloader via function
    train_dataloader() and val_dataloader().

    dataloader is an iterator that produces a tuple of (a list of raw text, 1D tensor label list).

    """

    def __init__(
        self,
        text_data_name,
        tokenizer_name,
        truncate: int = 256,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.truncate = truncate
        self.setup(text_data_name)

    def setup(self, text_data_name):

        self.train_data = TextData(text_data_name, "train")
        self.val_data = TextData(text_data_name, "test")
        self.build_vocab_and_label()
        self.label_pipeline = lambda x: self.label_map[x]

    def build_vocab_and_label(self):

        self.label_set = set()

        for (line, label) in self.train_data:
            self.label_set.add(label)
        self.label_map = dict(zip(self.label_set, range(len(self.label_set))))

    @property
    def label_size(self):
        return len(self.label_set)

    def get_collate(self):

        return partial(
            text_collate_with_tokenization,
            tokenizer=self.tokenizer,
            label_pipeline=self.label_pipeline,
            truncate=self.truncate,
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.get_collate(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.get_collate(),
        )


class NgramTextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        text_data_name,
        ngram_min: int = 1,
        ngram_max: int = 1,
        vocab_size: int = 50000,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.vocab_size = vocab_size
        self.setup(text_data_name)

    def setup(self, text_data_name):

        self.train_data = TextData(text_data_name, "train")
        self.val_data = TextData(text_data_name, "test")
        self.build_vocab_and_label()
        self.label_pipeline = lambda x: self.label_map[x]
        self.text_pipeline = lambda x: np.array(
            self.vectorizer.transform([x]).todense()
        )[0]

    def build_vocab_and_label(self):

        self.label_set = set()
        text_data_all = []
        for (line, label) in self.train_data:
            self.label_set.add(label)
            text_data_all.append(line)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(self.ngram_min, self.ngram_max), max_features=self.vocab_size
        ).fit(text_data_all)
        self.label_map = dict(zip(self.label_set, range(len(self.label_set))))

    # @property
    # def vocab_size(self):
    #     return len(self.vectorizer.get_feature_names_out())

    @property
    def label_size(self):
        return len(self.label_set)

    def get_collate(self):

        return partial(
            ngram_collate_fn,
            label_pipeline=self.label_pipeline,
            text_pipeline=self.text_pipeline,
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.get_collate(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.get_collate(),
        )


def _test_tokenized_text_data_module(return_length=False):
    data_module = TokenizedTextDataModule("AG_NEWS", return_length=return_length)

    train_dataloader = data_module.train_dataloader()

    print(data_module.vocab_size, data_module.label_size)

    for batch in train_dataloader:
        print(batch)
        break


def _test_text_data_module():
    data_module = TextDataModule("AG_NEWS")

    train_dataloader = data_module.train_dataloader()

    print(data_module.label_size)

    for batch in train_dataloader:
        print(batch)
        break


def _test_pretrained_tokenized_text_data_module():
    data_module = PretrainedTokenizedTextDataModule("AG_NEWS", "bert-base-uncased", 128)

    train_dataloader = data_module.train_dataloader()

    print(data_module.label_size)

    for batch in train_dataloader:
        print(batch)
        break


def _test_ngram_text_data_module():
    data_module = NgramTextDataModule("AG_NEWS")

    train_dataloader = data_module.train_dataloader()

    print(data_module.vocab_size, data_module.label_size)

    for batch in train_dataloader:
        print(batch)
        break


if __name__ == "__main__":
    # _test_tokenized_text_data_module(return_length=False)
    # _test_tokenized_text_data_module(return_length=True)
    _test_pretrained_tokenized_text_data_module()
    # _test_text_data_module()
    # _test_ngram_text_data_module()

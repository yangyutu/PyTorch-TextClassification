import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchtext import datasets
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import gensim
import torchtext


def get_data_iter(name, partition):
    if name == "AG_NEWS":
        data_iter = datasets.AG_NEWS(split=partition)
    elif name == "YelpReviewFull":
        data_iter = datasets.YelpReviewFull(split=partition)
    elif name == "IMDB":
        data_iter = datasets.IMDB(split=partition)
    else:
        raise NotImplementedError
    return data_iter


class CharDataset(Dataset):
    label_2_idx = {}
    num_classes = 0

    def __init__(self, name, alphabet, partition, max_len=1014):
        self.name = name
        self.partition = partition
        self.alphabet = alphabet
        self.alphaset_2_idx = dict(zip(alphabet, range(len(alphabet))))
        self.max_len = max_len
        self._setup()

    def _setup(self):
        data_iter = get_data_iter(self.name, self.partition)

        x_raw = []
        y_raw = []

        for label, line in data_iter:
            x_raw.append(line.lower())
            y_raw.append(label)

        def _convert(x):
            out = []
            for c in x:
                out.append(self.alphaset_2_idx.get(c, len(self.alphabet)))
            return out[: self.max_len]

        self.x_char = list(map(_convert, x_raw))
        if not CharDataset.label_2_idx:
            self._construct_label_2_idx_map(y_raw)
        self.y = [CharDataset.label_2_idx[y] for y in y_raw]

        x_char_tensor = [
            F.one_hot(torch.LongTensor(x), len(self.alphabet) + 1) for x in self.x_char
        ]

        self.x_char_tensor_full = []
        for x in x_char_tensor:
            full = torch.zeros((self.max_len, len(self.alphabet) + 1))
            full[: len(x), :] = x
            self.x_char_tensor_full.append(full.transpose(0, 1))

    def _construct_label_2_idx_map(self, y_raw):
        unique_y = set(y_raw)
        CharDataset.label_2_idx = dict(zip(unique_y, range(len(unique_y))))
        CharDataset.num_classes = len(unique_y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_char_tensor_full[idx], self.y[idx]


class TokenDataset(Dataset):
    label_2_idx = {}
    num_classes = 0
    vocab = None
    unk_index = -1
    pad_index = -1

    def __init__(self, name, partition, max_len, min_require_freq=10):
        self.name = name
        self.partition = partition
        self.max_len = max_len
        self.min_require_freq = min_require_freq

        self._setup()

    def _setup(self):
        data_iter = get_data_iter(self.name, self.partition)

        x_tokens = []
        self.x_lens = []
        y_raw = []

        for label, line in data_iter:
            tokens = gensim.utils.simple_preprocess(line.lower())
            self.x_lens.append(len(tokens))
            x_tokens.append(tokens)
            y_raw.append(label)

        if TokenDataset.vocab is None:
            TokenDataset.vocab = torchtext.vocab.build_vocab_from_iterator(
                x_tokens, min_freq=self.min_require_freq, specials=["<unk>", "<pad>"]
            )
            TokenDataset.unk_index = TokenDataset.vocab["<unk>"]
            TokenDataset.pad_index = TokenDataset.vocab["<pad>"]
            TokenDataset.vocab.set_default_index(TokenDataset.unk_index)

        x_ids = list(map(TokenDataset.vocab.forward, x_tokens))
        self.x_ids_pad = []
        for x in x_ids:
            if len(x) > self.max_len:
                self.x_ids_pad.append(x[: self.max_len])
            else:
                self.x_ids_pad.append(
                    x + [TokenDataset.pad_index] * (self.max_len - len(x))
                )

        if not TokenDataset.label_2_idx:
            self._construct_label_2_idx_map(y_raw)
        self.y = [TokenDataset.label_2_idx[y] for y in y_raw]

    def _construct_label_2_idx_map(self, y_raw):
        unique_y = set(y_raw)
        TokenDataset.label_2_idx = dict(zip(unique_y, range(len(unique_y))))
        TokenDataset.num_classes = len(unique_y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x_ids_pad[idx], self.x_lens[idx]), self.y[idx]


class BOWDataset(Dataset):
    # Bag of words (include n-gram) dataset
    label_2_idx = {}
    num_classes = 0
    count_vectorizer = None
    tfidf_transformer = None

    def __init__(self, name, partition, most_frequent_words, ngram_range, tf_idf):
        self.name = name
        self.partition = partition
        self.most_frequent_words = most_frequent_words
        self.ngram_range = ngram_range
        self.tf_idf = tf_idf

        self._setup()

    def _setup(self):
        data_iter = get_data_iter(self.name, self.partition)

        x_raw = []
        y_raw = []

        for label, line in data_iter:
            x_raw.append(line.lower())
            y_raw.append(label)

        if BOWDataset.count_vectorizer is None:
            BOWDataset.count_vectorizer = CountVectorizer(
                ngram_range=self.ngram_range, max_features=self.most_frequent_words
            )
            BOWDataset.count_vectorizer.fit(x_raw)

        x_count = BOWDataset.count_vectorizer.transform(x_raw)
        if self.tf_idf:
            if BOWDataset.tfidf_transformer is None:
                BOWDataset.tfidf_transformer = TfidfTransformer()
                BOWDataset.tfidf_transformer.fit(x_count)

            x_count = BOWDataset.tfidf_transformer.transform(x_count)

        if not BOWDataset.label_2_idx:
            self._construct_label_2_idx_map(y_raw)
        self.y = [BOWDataset.label_2_idx[y] for y in y_raw]

        # self.x_count = [torch.FloatTensor(x.toarray()) for x in x_count]
        self.x_count = torch.FloatTensor(x_count.toarray())

    def _construct_label_2_idx_map(self, y_raw):
        unique_y = set(y_raw)
        BOWDataset.label_2_idx = dict(zip(unique_y, range(len(unique_y))))
        BOWDataset.num_classes = len(unique_y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_count[idx], self.y[idx]


if __name__ == "__main__":
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
    print(len(alphabet))
    # dataloader = TextClassicfiationDataModule('AG_NEWS', alphabet)

    # dataloader.prepare_data()

    # train_dataset = CharDataset('AG_NEWS', alphabet, 'train')

    # train_len = int(0.9 * len(train_dataset))
    # train, val = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    # print(len(train))
    # print(len(val))

    # test_dataset = CharDataset('AG_NEWS', alphabet, 'test')

    # print(len(test_dataset))
    # #print(train[0].size())
    # x, y = test_dataset[0]

    # test_dataset = BOWDataset(
    #     "IMDB", "test", most_frequent_words=50000, ngram_range=(1, 1), tf_idf=True
    # )

    # x, y = test_dataset[0]
    # print(x, y)
    # # print(torch.sum(x))

    # count_vector = BOWDataset.count_vectorizer

    # vocab = count_vector.vocabulary_

    # print(vocab)

    test_dataset = TokenDataset("IMDB", "test", 256, 10)
    x, y = test_dataset[0]
    print(x, y)


from torchtext import datasets
import torchtext
import gensim
import gensim.downloader

train_iter = datasets.IMDB(split="test")

print(len(train_iter))
for label, line in train_iter:
    # print(label, line)

    break


def get_tokens(data_iter):
    for label, line in data_iter:
        tokens = gensim.utils.simple_preprocess(line)
        yield tokens


vocab = torchtext.vocab.build_vocab_from_iterator(
    get_tokens(train_iter), min_freq=10, specials=["<unk>", "<pad>"]
)
unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]
vocab.set_default_index(unk_index)
# nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
# tokenizer = nlp.tokenizer
# print(tokenizer(line))
# doc = nlp(line)
# for token in doc:
#     print(token.text)

# glove = torchtext.vocab.GloVe(name="840B", dim=300)


# glove_vectors = gensim.downloader.load("glove-twitter-25")
print(len(vocab))
print(vocab["<unk>"])
print(vocab.get_itos()[:100])

print(vocab.forward(["oiq"]))

print(vocab.forward(gensim.utils.simple_preprocess(line)))
glove = torchtext.vocab.GloVe(name="6B", dim=300)

embeddings_vocab = glove.get_vecs_by_tokens(vocab.get_itos())
print(embeddings_vocab.size())
print("done")


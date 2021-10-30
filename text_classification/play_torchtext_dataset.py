from torchtext import datasets

train_iter = datasets.IMDB(split='train')

print(len(train_iter))
for label, line in train_iter:
    print(label, line)
    break 

train_iter = datasets.AG_NEWS(split='train')

for label, line in train_iter:
    print(label, line)
    break 

train_iter = datasets.YelpReviewFull(split='train')

for label, line in train_iter:
    print(label, line)
    break 



train_iter = datasets.YahooAnswers(split='train')

for label, line in train_iter:
    print(label, line)
    break 



# train_iter = datasets.AG_NEWS(split='train')

# print(train_iter)
from typing import Text
from models.text_classifier import CharacterLevelCNN, TextClassifier, LogisticBOWModel
from data_loader import BOWDataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

dataset_name = "AG_NEWS"

# YelpReviewFull AG_NEWS IMDB


def get_dataloaders(name, most_frequent_words, ngram_range, tf_idf):

    train_dataset = BOWDataset(name, "train", most_frequent_words, ngram_range, tf_idf)
    test_dataset = BOWDataset(name, "test", most_frequent_words, ngram_range, tf_idf)
    train_len = int(0.9 * len(train_dataset))
    train, val = random_split(
        train_dataset, [train_len, len(train_dataset) - train_len]
    )

    train_dataset_loader = DataLoader(train, batch_size=128, num_workers=16)

    val_dataset_loader = DataLoader(val, batch_size=128, num_workers=16)

    test_dataset_loader = DataLoader(test_dataset, batch_size=128, num_workers=16)

    num_classes = test_dataset.num_classes
    return train_dataset_loader, val_dataset_loader, test_dataset_loader, num_classes


most_frequent_words = 50000
ngram_range = (1, 5)
tf_idf = True
(
    train_dataset_loader,
    val_dataset_loader,
    test_dataset_loader,
    num_classes,
) = get_dataloaders(dataset_name, most_frequent_words, ngram_range, tf_idf)

print(f"num of classes: {num_classes}")
config = {}

config["num_classes"] = num_classes
config["most_frequent_words"] = most_frequent_words
config["lr"] = 0.001
model_BOW = LogisticBOWModel(config)
model = TextClassifier(config, model=model_BOW)

model_name = f"BOW_{ngram_range}_idf_{tf_idf}"

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"./{model_name}_{dataset_name}/",
    filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)
lr_monitor = LearningRateMonitor(logging_interval="step")

project_name = "text_classification"
wandb_logger = WandbLogger(
    project=project_name,  # group runs in "MNIST" project
    log_model="all",
    save_dir="./results/logs/",
)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=3,
    logger=wandb_logger,
    callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback],
)


print(dataset_name)

trainer.fit(
    model,
    train_dataloaders=train_dataset_loader,
    val_dataloaders=val_dataset_loader,
)

# ckpt = "/home/ubuntu/MLData/work/Repos/pytorch-examples/text_classification/lightning_logs/version_5/checkpoints/epoch=199-step=168799.ckpt"
# load_model = CharacterLevelCNN.load_from_checkpoint(ckpt, config=config)
trainer.test(model, dataloaders=test_dataset_loader, ckpt_path="best")

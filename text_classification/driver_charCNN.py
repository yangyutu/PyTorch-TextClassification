from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from models.text_classifier import CharacterLevelCNN, GenericModel

max_seq_length = 1014
# max_seq_length = 100
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"

dataset_name = "AG_NEWS"

# YelpReviewFull AG_NEWS IMDB


def get_dataloaders(name, alphabet, max_seq_length):

    train_dataset = CharDataset(name, alphabet, "train", max_len=max_seq_length)
    test_dataset = CharDataset(name, alphabet, "test", max_len=max_seq_length)
    train_len = int(0.9 * len(train_dataset))
    train, val = random_split(
        train_dataset, [train_len, len(train_dataset) - train_len]
    )

    train_dataset_loader = DataLoader(train, batch_size=128, num_workers=16)

    val_dataset_loader = DataLoader(val, batch_size=128, num_workers=16)

    test_dataset_loader = DataLoader(test_dataset, batch_size=128, num_workers=16)

    num_classes = test_dataset.num_classes
    return train_dataset_loader, val_dataset_loader, test_dataset_loader, num_classes


(
    train_dataset_loader,
    val_dataset_loader,
    test_dataset_loader,
    num_classes,
) = get_dataloaders(dataset_name, alphabet, max_seq_length)
config = {}

config["num_classes"] = num_classes
config["num_characters"] = len(alphabet) + 1
config["max_seq_length"] = max_seq_length
config["dropout_input"] = 0.1
config["lr"] = 0.001
model_charcnn = CharacterLevelCNN(config)
model = GenericModel(config, model_charcnn)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"./cnn_char_{dataset_name}/",
    filename="cnn_char-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

trainer = Trainer(
    gpus=1,
    auto_select_gpus=True,
    max_epochs=3,
    progress_bar_refresh_rate=20,  # disable progress bar by setting to 0
    logger=TensorBoardLogger("lightning_logs/", name=f"cnn_char_{dataset_name}"),
    callbacks=[checkpoint_callback],
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

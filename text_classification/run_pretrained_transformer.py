from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger

from data_utils.text_data import TextDataModule
from models.text_classifier import TextClassifier
from models.pretrained_transformers import PretrainedBertEncoder
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):

    seed_everything(args.seed)

    dataset_name = args.dataset_name
    # dataset names should be AG_NEWS, IMDB

    data_module = TextDataModule(
        dataset_name, batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    config = {}
    config["num_classes"] = data_module.label_size
    config["lr"] = args.lr
    encoder_model = PretrainedBertEncoder(
        pretrained_model_name=args.pretrained_model_name,
        num_classes=data_module.label_size,
        truncate=args.truncate,
    )
    model = TextClassifier(config, model=encoder_model)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc_epoch",
        save_top_k=3,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    project_name = "text_classification"
    wandb_logger = WandbLogger(
        project=project_name,  # group runs in "MNIST" project
        log_model="all" if args.log_model else False,
        save_dir=args.default_root_dir,
        group=args.dataset_name,
        tags=[args.pretrained_model_name, args.dataset_name],
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate),
            checkpoint_callback,
            lr_monitor,
        ],
        deterministic=True,
    )

    print(dataset_name)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def parse_arguments():

    parser = ArgumentParser()

    # trainer specific arguments

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, required=True)

    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=5)

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--log_model", action="store_true")

    # model specific arguments
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--truncate", type=int, default=128)
    parser.add_argument("--pretrained_model_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    main(args)

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb

import argparse

from ship_segmentation.models.unet import UNetLitModel
from ship_segmentation.data.dataset import ImagesDataset
from ship_segmentation.data.utils import get_mask


def get_datasets(metadata_df, images_dir, get_mask):
    unique_img_ids = metadata_df["ImageId"].unique()
    np.random.shuffle(unique_img_ids)

    train_ids = unique_img_ids[: int(len(unique_img_ids) * 0.8)]
    val_ids = unique_img_ids[
        int(len(unique_img_ids) * 0.8) : int(len(unique_img_ids) * 1.0)
    ]

    train_metadata = metadata_df[metadata_df["ImageId"].isin(train_ids)].reset_index(
        drop=True
    )
    val_metadata = metadata_df[metadata_df["ImageId"].isin(val_ids)].reset_index(
        drop=True
    )

    train_ds = ImagesDataset(images_dir, train_metadata, get_mask)
    val_ds = ImagesDataset(images_dir, val_metadata, get_mask)

    return train_ds, val_ds


def get_input_transform():
    transform = A.Compose(
        [
            A.Flip(),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.Resize(height=384, width=384),  # 224, 576
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    return transform


def get_dataloaders(train_ds, val_ds, input_transform, batch_size=16):
    def collate(batch):
        imgs, masks = [], []
        for img, mask in batch:
            augs = input_transform(image=img, mask=mask)
            imgs.append(augs["image"][None])
            masks.append(augs["mask"][None])

        imgs = torch.cat(imgs).float()
        masks = torch.cat(masks).long()

        return imgs, masks

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        num_workers=3,
    )
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=3,
    )

    return train_dl, val_dl


def main():
    # to reproduce experiments
    np.random.seed(0)

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")
    parser.add_argument(
        "--device_number", type=int, default=1, help="GPU device number"
    )
    parser.add_argument("--output_path", default="output")
    parser.add_argument(
        "--continue_train",
        action="store_true",
        help="Continue training from checkpoint",
    )
    parser.add_argument("--ckpt_path", default="output/ckpts/last.ckpt")

    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb for logging"
    )
    parser.add_argument("--wandb_key", default="your_key")
    parser.add_argument("--wandb_project", default="ship_segmentation")
    parser.add_argument("--wandb_entity", default="viktor_povazhuk")

    parser.add_argument(
        "--images_path",
        default="/home/viktor/PythonProjects/data/airbus-ship-detection/train_v2",
        help="Path to the folder with images.",
    )
    parser.add_argument(
        "--metadata_path",
        default="/home/viktor/PythonProjects/data/airbus-ship-detection/train_ship_segmentations_v2.csv",
        help="Path to the metadata csv file.",
    )

    parser.add_argument("--limit_train_batches", default=500, type=int)
    parser.add_argument("--limit_val_batches", default=100, type=int)

    args = parser.parse_args()

    output_path = Path(args.output_path)
    images_dir = Path(args.images_path)

    metadata = pd.read_csv(args.metadata_path)
    train_ds, val_ds = get_datasets(metadata, images_dir, get_mask=get_mask)
    input_transform = get_input_transform()
    train_dl, val_dl = get_dataloaders(
        train_ds, val_ds, input_transform, batch_size=args.batch_size
    )

    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_number}")
        else:
            print("GPU is not available. Default to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model = UNetLitModel(args.learning_rate)

    if args.use_wandb:
        wandb.login(key=args.wandb_key)
        logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity)
        logger.experiment.config.update(
            {
                "device": device,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "continue": args.continue_train,
            }
        )
    else:
        logger = TensorBoardLogger("lightning_logs")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-5,
        patience=4,
        verbose=True,
        mode="min",
        strict=True,
    )
    latest_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="global_step",
        mode="max",
        dirpath=str(output_path / "ckpts"),
        filename="last",
    )
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_dice",
        mode="max",
        dirpath=str(output_path / "ckpts"),
        filename="{val_dice:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=device.type,
        devices=args.device_number,
        logger=logger,
        callbacks=[
            early_stop_callback,
            latest_checkpoint_callback,
            best_checkpoint_callback,
        ],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    if args.continue_train:
        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=args.ckpt_path,
        )
    else:
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    model.eval()

    dl_iter = iter(val_dl)

    for n in range(4):
        images, target = next(dl_iter)

        target = target.long()

        with torch.no_grad():
            output = model(images)

        pred = torch.max(output, 1)[1]

        target = target.numpy()
        pred = pred.numpy()

        fig = plt.figure(figsize=(6, 3 * args.batch_size))
        for i in range(args.batch_size):
            ax = fig.add_subplot(args.batch_size, 2, 2 * i + 1, xticks=[], yticks=[])
            ax.imshow(pred[i])
            ax.set_title(f"pred [{i}]", fontsize=10)
            ax = fig.add_subplot(args.batch_size, 2, 2 * i + 2, xticks=[], yticks=[])
            ax.imshow(target[i])
            ax.set_title(f"target [{i}]", fontsize=10)

        plt.savefig(output_path / f"predictions_{n+1}.png")


if __name__ == "__main__":
    main()
    print("Training is finished")

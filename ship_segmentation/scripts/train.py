from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from torchmetrics import Dice

import pytorch_lightning as pl
from torchvision.models import vgg11, VGG11_Weights

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb

import argparse


# convert run length encoding to image mask
def get_mask(img, metadata_df, img_id):
    h, w = img.shape[:2]
    mask = np.zeros(img.shape[:2]).astype(int)

    # need to get all corresponding rows because each row is a mask of one object
    encoded_pixels_list = list(
        metadata_df[metadata_df["ImageId"] == img_id]["EncodedPixels"]
    )
    for encoded_pixels in encoded_pixels_list:
        # check if it is float because otherwise throw exception
        if isinstance(encoded_pixels, float) and np.isnan(encoded_pixels):
            continue
        encoded_pixels = list(map(int, encoded_pixels.split(" ")))
        for i in range(int(len(encoded_pixels) / 2)):
            ran = np.arange(encoded_pixels[i * 2 + 1])
            ran = ran + encoded_pixels[i * 2]
            for num in ran:
                num = num - 1
                col = num // h
                row = num - h * col
                mask[row][col] = 1
    return mask


class ImagesDataset(Dataset):
    def __init__(self, images_dir, metadata_df, get_mask):
        self.images_dir = images_dir
        self.metadata_df = metadata_df
        self.image_ids = metadata_df["ImageId"].unique()
        self.get_mask = get_mask

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        img = cv2.imread(str(self.images_dir / img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self.get_mask(img, self.metadata_df, img_id)

        return img, mask


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


def up_conv(in_channels, out_channels):
    mod = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
    )
    return mod


def conv(in_channels, out_channels):
    mod = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
    )
    return mod


# realization of the paper: https://arxiv.org/pdf/1801.05746.pdf
class UNetVGG11Model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = VGG11_Weights.DEFAULT
        else:
            weights = None
        encoder_layers = vgg11(weights=weights, progress=True).features

        self.block1 = nn.Sequential(*encoder_layers[:2])
        self.block2 = nn.Sequential(*encoder_layers[2:5])
        self.block3 = nn.Sequential(*encoder_layers[5:10])
        self.block4 = nn.Sequential(*encoder_layers[10:15])
        self.block5 = nn.Sequential(*encoder_layers[15:20])

        self.bottleneck = nn.Sequential(
            encoder_layers[20],
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.up_conv1 = up_conv(512, 256)
        self.conv1 = conv(256 + 512, 512)
        self.up_conv2 = up_conv(512, 256)
        self.conv2 = conv(256 + 512, 512)
        self.up_conv3 = up_conv(512, 128)
        self.conv3 = conv(128 + 256, 256)
        self.up_conv4 = up_conv(256, 64)
        self.conv4 = conv(64 + 128, 128)
        self.up_conv5 = up_conv(128, 32)

        self.mapper = nn.Conv2d(96, 2, kernel_size=1, stride=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.bottleneck(block5)

        x = self.up_conv1(x)
        x = torch.cat((block5, x), dim=1)
        x = self.conv1(x)
        x = self.up_conv2(x)
        x = torch.cat((block4, x), dim=1)
        x = self.conv2(x)
        x = self.up_conv3(x)
        x = torch.cat((block3, x), dim=1)
        x = self.conv3(x)
        x = self.up_conv4(x)
        x = torch.cat((block2, x), dim=1)
        x = self.conv4(x)
        x = self.up_conv5(x)
        x = torch.cat((block1, x), dim=1)

        x = self.mapper(x)

        return x


class UNetLitModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()

        self.model = UNetVGG11Model()
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.dice_scorer = Dice(average="micro", ignore_index=0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.trainer.global_step == 0 and wandb.run is not None:
            wandb.define_metric("train_loss", summary="mean")
            wandb.define_metric("train_dice", summary="mean")
            wandb.define_metric("val_loss", summary="mean")
            wandb.define_metric("val_dice", summary="mean")
            wandb.define_metric("test_loss", summary="mean")
            wandb.define_metric("test_dice", summary="mean")
        x, y = batch
        y = y.type(torch.LongTensor).to(self.device)
        output = self(x)
        loss = self.criterion(output, y)
        dice = self.dice_scorer(output, y)
        self.log("train_loss", loss)
        self.log("train_dice", dice)
        self.log(
            "global_step", torch.tensor(self.trainer.global_step, dtype=torch.float32)
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.LongTensor).to(self.device)
        output = self(x)
        loss = self.criterion(output, y)
        dice = self.dice_scorer(output, y)
        self.log("val_loss", loss)
        self.log("val_dice", dice)
        self.log(
            "global_step", torch.tensor(self.trainer.global_step, dtype=torch.float32)
        )
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.LongTensor).to(self.device)
        output = self(x)
        loss = self.criterion(output, y)
        dice = self.dice_scorer(output, y)
        self.log("test_loss", loss)
        self.log("test_dice", dice)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            threshold=1e-3,
            threshold_mode="abs",
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--device_number", type=int, default=1, help="GPU device number")
    parser.add_argument("--output_path", default="output")
    parser.add_argument("--continue_train", action="store_true", help="Continue training from checkpoint")
    parser.add_argument("--ckpt_path", default="output/ckpts/last.ckpt")

    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_key", default="your_key")
    parser.add_argument("--wandb_project", default="ship_segmentation")
    parser.add_argument("--wandb_entity", default="viktor_povazhuk")

    parser.add_argument(
        "--images_path",
        default="/home/viktor/PythonProjects/data/airbus-ship-detection/train_v2",
        help="Path to the folder with images."
    )
    parser.add_argument(
        "--metadata_path",
        default="/home/viktor/PythonProjects/data/airbus-ship-detection/train_ship_segmentations_v2.csv",
        help="Path to the metadata csv file."
    )

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
        monitor="val_dice",
        min_delta=1e-3,
        patience=4,
        verbose=True,
        mode="max",
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
        limit_train_batches=500,
        limit_val_batches=100,
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

    for _ in range(4):
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

        plt.savefig(output_path / f"predictions_{i+1}.png")


if __name__ == "__main__":
    main()
    print("Training is finished")

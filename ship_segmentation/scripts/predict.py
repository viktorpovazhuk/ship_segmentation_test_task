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

import time


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

    parser.add_argument("--image_path", type=str, required=True)
    # TODO: implement thresholding
    parser.add_argument("--cls_threshold", type=float, default=0.3)

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--device_number", type=int, default=1)

    parser.add_argument("--output_path", default="output/predictions/")

    parser.add_argument("--ckpt_path", default="output/ckpts/last.ckpt")

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_number}")
        else:
            print("GPU is not available. Default to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model = UNetLitModel.load_from_checkpoint(args.ckpt_path).to(device)

    img = cv2.imread(args.image_path)

    transform = A.Compose(
        [
            A.Resize(height=384, width=384),  # 224, 576
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    transformed_img = transform(image=img)["image"]
    transformed_img = transformed_img.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(transformed_img)

    pred = torch.max(output, 1)[1].cpu().numpy()

    fig = plt.figure(figsize=(6, 3))

    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    ax.imshow(img)
    ax.set_title(f"img", fontsize=10)
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    ax.imshow(pred[0])
    ax.set_title(f"pred", fontsize=10)

    plt.savefig(output_path / f"pred_{time.time()}.png")


if __name__ == "__main__":
    main()

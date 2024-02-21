import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from torchmetrics import Dice

import pytorch_lightning as pl
from torchvision.models import vgg11, VGG11_Weights

import wandb

from ship_segmentation.models.blocks import up_conv, conv
from ship_segmentation.losses.dice_loss import DiceLoss


LOSSES = {
    "dice": DiceLoss,
    "ce": nn.CrossEntropyLoss,
}


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
    def __init__(self, learning_rate=1e-3, loss="dice"):
        super().__init__()

        self.model = UNetVGG11Model()
        self.learning_rate = learning_rate
        self.criterion = LOSSES[loss]()
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

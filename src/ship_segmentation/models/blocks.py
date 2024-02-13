import torch.nn as nn


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

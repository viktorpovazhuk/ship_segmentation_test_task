import torch.nn as nn
import torch.nn.functional as F


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
class DiceLoss(nn.Module):
    """Accepts logits as inputs."""

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim=1)[:, 1, ...]

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

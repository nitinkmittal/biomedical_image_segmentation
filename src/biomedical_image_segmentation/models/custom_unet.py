import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import CenterCrop
from typing import Union, List

from .unet import *


class CustomUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()

        self.conv_2d = conv2d_block(in_channels=64, out_channels=64)
        self.last_conv2d = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):

        x, down_X = self.encoder(x)
        x = self.decoder(x, down_X)
        x = self.conv_2d(x)
        return self.last_conv2d(x)

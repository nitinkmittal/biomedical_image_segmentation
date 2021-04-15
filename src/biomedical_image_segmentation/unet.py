import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import CenterCrop
from typing import Union, List


def conv2d_block(in_channels: int, out_channels: int):
    """Convolution block."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        ),
        nn.ReLU(),
    )


class UnetEncoder(nn.Module):
    """Encoder for UNet model."""

    def __init__(self):
        super().__init__()

        # why moduleList: https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
        self.down = nn.ModuleList()
        self.down.append(conv2d_block(1, 64))
        self.down.append(conv2d_block(64, 128))
        self.down.append(conv2d_block(128, 256))
        self.down.append(conv2d_block(256, 512))
        self.down.append(conv2d_block(512, 1024))

        self.depth = len(self.down)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        down_X = []
        for i, layer in enumerate(self.down):
            x = layer(x)
            if i < self.depth - 1:
                down_X.append(x)
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        return x, down_X


def conv2d_up_block(in_channels: int, out_channels: int):
    """Up Convolution block for UNet."""
    # https://towardsdatascience.com/is-the-transposed-convolution-layer-and-convolution-layer-the-same-thing-8655b751c3a1
    return nn.ModuleList(
        [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            conv2d_block(in_channels=in_channels, out_channels=out_channels),
        ]
    )


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # why moduleList: https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
        self.up = nn.ModuleList()
        self.up.append(conv2d_up_block(1024, 512))
        self.up.append(conv2d_up_block(512, 256))
        self.up.append(conv2d_up_block(256, 128))
        self.up.append(conv2d_up_block(128, 64))

    def forward(
        self, x: torch.Tensor, down_X: List[torch.Tensor]
    ) -> torch.Tensor:
        for i, layer in enumerate(self.up):
            x = layer[0](x)
            h, w = x.size()[-2:]
            x = torch.cat([CenterCrop(h)(down_X[-(i + 1)]), x], dim=1)
            x = layer[1](x)
        return x

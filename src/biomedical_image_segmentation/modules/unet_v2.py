from typing import List, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import CenterCrop
from .modules import activations, norm2d


class DoubleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        padding_mode: str,
        norm: str,
        act: str,
    ):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            norm2d(norm, num_features=out_channels),
            activations(act),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            norm2d(norm, num_features=out_channels),
            activations(act),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: int,
        padding: str,
        padding_mode: str,
        norm: str,
        act: str,
    ):
        super().__init__()
        self.downs = nn.ModuleList()

        for i, _ in enumerate(out_channels):

            self.downs.append(
                DoubleConv2d(
                    in_channels=out_channels[i - 1] if i > 0 else in_channels,
                    out_channels=out_channels[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    norm=norm,
                    act=act,
                )
            )

        self.depth = len(self.downs)

    def forward(self, x) -> Tuple[Tensor, List[Tensor]]:

        downs_X = []
        for i, layer in enumerate(self.downs):
            x = layer(x)
            if i < self.depth - 1:
                downs_X.append(x)
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, downs_X


class UpSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str,
        padding_mode: str,
        norm: str,
        act: str,
    ):
        super().__init__()
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                ),
                DoubleConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    norm=norm,
                    act=act,
                ),
            ]
        )

    def forward(self, x1: Tensor, downs_x: Tensor) -> Tensor:
        x1 = self.up[0](x1)
        h, _ = x1.size()[-2:]
        return self.up[1](torch.cat([CenterCrop(h)(downs_x), x1], dim=1))


class UNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: int,
        padding: str,
        padding_mode: str,
        norm: str,
        act: str,
    ):
        super().__init__()

        self.ups = nn.ModuleList()
        for i, _ in enumerate(out_channels):
            self.ups.append(
                UpSample(
                    in_channels=out_channels[i - 1] if i > 0 else in_channels,
                    out_channels=out_channels[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    norm=norm,
                    act=act,
                )
            )

    def forward(self, x1: Tensor, downs_X: List[Tensor]) -> torch.Tensor:
        for i, layer in enumerate(self.ups):
            x1 = layer(x1, downs_X[-(i + 1)])
        return x1


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: int,
        padding: str,
        padding_mode: str,
        norm: str,
        act: str,
    ):
        super().__init__()

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            norm=norm,
            act=act,
        )
        self.decoder = UNetDecoder(
            in_channels=out_channels[-1],
            out_channels=out_channels[::-1][1:],
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            norm=norm,
            act=act,
        )

    def forward(self, x) -> Tensor:

        x1, downs_X = self.encoder(x)
        x1 = self.decoder(x1, downs_X)

        return x1

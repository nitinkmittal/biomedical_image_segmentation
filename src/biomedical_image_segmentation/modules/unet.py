from typing import List, Tuple, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import CenterCrop
from .modules import activations, norm2d

from biomedical_image_segmentation.utils import copy_docstring


class MultiConv2d(nn.Module):
    """Mutli-2D convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        padding_mode: str,
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        bias: bool,
        norm: str,
        act: str,
        num_conv2d: int,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input image

            out_channels (int): Number of channels produced by the convolution

            kernel_size (int or tuple): Size of the convolving kernel

            stride (int or tuple): Stride of the convolution

            padding (int, tuple or str): Padding added to all four sides of the input

            padding_mode (string): 'zeros', 'reflect', 'replicate' or 'circular'

            dilation (int or tuple): Spacing between kernel elements

            bias (bool): If True, adds a learnable bias to the output

            norm (str): type of normalization post convolution operation

            act (str): type of activation post normalization operation

            num_conv2d (int): number of (convolution + norm + act) layers
        """
        super().__init__()
        self.down = nn.ModuleList()
        for i in range(num_conv2d):
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        padding_mode=padding_mode,
                        dilation=dilation,
                        bias=bias,
                    ),
                    norm2d(norm, num_features=out_channels),
                    activations(act),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for Multi-2d convolution block.

        Args:
            x (tensor): shape (N, C_in, H_in, W_in)
                N: number of samples in a batch

                C_in: number of channels

                H_in: height of tensor

                W_in: width of tensor

        Returns:
            A tensor of shape (N, C_out, H_out, W_out)
        """
        for layer in self.down:
            x = layer(x)
        return x


class UNetEncoder(nn.Module):
    """Encoder for UNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        padding_mode: str,
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        bias: bool,
        norm: str,
        act: str,
        num_conv2d: int,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input image

            out_channels (sequence of integers): Sequence of number of channels
                produced by the convolution blocks

            kernel_size (int or tuple): Size of the convolving kernel

            stride (int or tuple): Stride of the convolution

            padding (int, tuple or str): Padding added to all four sides of the input

            padding_mode (string): 'zeros', 'reflect', 'replicate' or 'circular'

            dilation (int or tuple): Spacing between kernel elements

            bias (bool): If True, adds a learnable bias to the output

            norm (str): type of normalization post convolution operation

            act (str): type of activation post normalization operation

            num_conv2d (int): number of (convolution + norm + act) layers
        """
        super().__init__()
        self.downs = nn.ModuleList()

        for i, _ in enumerate(out_channels):
            self.downs.append(
                MultiConv2d(
                    in_channels=out_channels[i - 1] if i > 0 else in_channels,
                    out_channels=out_channels[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    norm=norm,
                    act=act,
                    num_conv2d=num_conv2d,
                )
            )
        self.depth = len(self.downs)

    def forward(self, x) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass for UNet encoder.

        Args:
            x (tensor): shape (N, C_in, H_in, W_in)
                N: number of samples in a batch

                C_in: number of channels

                H_in: height of tensor

                W_in: width of tensor

        Returns:
            output of encoding: A tensor of shape (N, C_out, H_out, W_out)

            down_X (list of tensors): residuals from subsequent pooling during encoding
        """
        down_X = []
        for i, layer in enumerate(self.downs):
            x = layer(x)
            if i < self.depth - 1:
                down_X.append(x.clone())
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, down_X


class UpSample(nn.Module):
    """Upsampling block for UNet decoder."""

    @copy_docstring(MultiConv2d.__init__)
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        padding_mode: str,
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        bias: bool,
        norm: str,
        act: str,
        num_conv2d: int,
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
                MultiConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    norm=norm,
                    act=act,
                    num_conv2d=num_conv2d,
                ),
            ]
        )

    def forward(self, x1: Tensor, down_x: Tensor) -> Tensor:
        """Forward pass for Upsampling block.

        Args:
            x1 (tensor): shape (N, C_in, H_in, W_in)
                N: number of samples in a batch

                C_in: number of channels

                H_in: height of tensor

                W_in: width of tensor

            down_x (tensor): shape (N, C_in, H_in, W_in)
                residual from pooling across same level during encoding

        Returns:
            output of upsampler: A tensor of shape (N, C_out, H_out, W_out)
        """
        x1 = self.up[0](x1)
        h, _ = x1.size()[-2:]
        return self.up[1](torch.cat([CenterCrop(h)(down_x), x1], dim=1))


class UNetDecoder(nn.Module):
    """Decoder for UNet."""

    @copy_docstring(UNetEncoder.__init__)
    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        padding_mode: str,
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        bias: bool,
        norm: str,
        act: str,
        num_conv2d: int,
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
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    norm=norm,
                    act=act,
                    num_conv2d=num_conv2d,
                )
            )

    def forward(self, x: Tensor, down_X: List[Tensor]) -> torch.Tensor:
        """Forward pass for UNet encoder.

        Args:
            x (tensor): shape (N, C_in, H_in, W_in)
                N: number of samples in a batch

                C_in: number of channels

                H_in: height of tensor

                W_in: width of tensor

            down_X (list of tensors): residuals from subsequent pooling during encoding

        Returns:
            output of decoder: A tensor of shape (N, C_out, H_out, W_out)
        """
        for i, layer in enumerate(self.ups):
            x = layer(x, down_X[-(i + 1)])
        return x


class UNet(nn.Module):
    """Customizable UNet architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
        padding_mode: str = "zeros",
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        norm: str = "identity",
        act: str = "identity",
        num_conv2d: int = 1,
    ):
        """
        Args:
            in_channels (int): Number of channels in the input image

            out_channels (sequence of integers): Sequence of number of channels
                produced by the convolution blocks

            kernel_size (int or tuple): Size of the convolving kernel

            stride (int or tuple): Stride of the convolution
                default: 1

            padding (int, tuple or str): Padding added to all four sides of the input
                default: 0

            padding_mode (string): 'zeros', 'reflect', 'replicate' or 'circular'
                default: 'zeros'

            dilation (int or tuple): Spacing between kernel elements
                default: 1

            bias (bool): If True, adds a learnable bias to the output
                default: True

            norm (str): type of normalization post convolution operation
                default: "identity"

            act (str): type of activation post normalization operation
                default: "identity"

            num_conv2d (int): number of (convolution + norm + act) layers
                default: 1
        """
        super().__init__()

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride=stride,
            dilation=dilation,
            bias=bias,
            norm=norm,
            act=act,
            num_conv2d=num_conv2d,
        )
        self.decoder = UNetDecoder(
            in_channels=out_channels[-1],
            out_channels=out_channels[::-1][1:],
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            stride=stride,
            dilation=dilation,
            bias=bias,
            norm=norm,
            act=act,
            num_conv2d=num_conv2d,
        )

    def forward(self, x) -> Tensor:
        """Forward pass for UNet.

        Args:
            x (tensor): shape (N, C_in, H_in, W_in)
                N: number of samples in a batch

                C_in: number of channels

                H_in: height of tensor

                W_in: width of tensor

        Returns:
            A tensor of shape (N, C_out, H_out, W_out)
        """
        x, down_X = self.encoder(x)
        x = self.decoder(x, down_X)
        return x

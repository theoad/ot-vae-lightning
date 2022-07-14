"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a modular CNN

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from torch import Tensor
from typing import Union, List, Optional
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from math import sqrt, log2
import numpy as np
from sympy import divisors


# ******************************************************************************************************************** #


class ConvLayer(nn.Conv2d):
    def __init__(
            self,
            in_features: int,
            out_features: int,

            # ConvLayer params
            residual: bool = False,
            normalization: Optional[str] = None,
            activation: Optional[str] = None,
            equalizer_lr: bool = False,
            dropout: float = 0.,

            # nn.Conv2d params
            kernel_size: _size_2_t = 3,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 1,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
    ) -> None:
        """
        nn.Conv2 Layer that supports:
            -   residual skip connections
            -   normalization
            -   activation
            -   `equalized learning rate <https://arxiv.org/abs/1710.10196>`_

        :param in_features: Number of channels in the input image
        :param out_features: Number of channels produced by the convolution

        :param bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        :param residual: If ``True``, add the input to the output tensor (skip connection). Default: ``False``
        :param normalization: Normalization layer. nn.Module expecting parameter ``num_features`` in `__init__`.
                              Default None
        :param activation: Activation layer. nn.Module expecting no parameter in ``__init__``. Default `None
        :param equalizer_lr: If ``True`` normalize the convolution weights by (in_features * kernel_size ** 2).
                             Default ``False``
        :param dropout: Dropout probability

        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param padding: Padding added to all four sides of the input. Default: 0
        :param dilation: Spacing between kernel elements. Default: 1
        :param groups:  Number of blocked connections from input channels to output channels. Default: 1
        """
        super().__init__(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias)

        self.residual = residual
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1 / (in_features * kernel_size ** 2) if equalizer_lr else 1

        if normalization is None or "none" in normalization.lower() or "null" in normalization.lower():
            self.normalization = nn.Identity()
        elif "batch" in normalization.lower():
            self.normalization = nn.BatchNorm2d(in_features)
        elif "group" in normalization.lower():
            self.normalization = nn.GroupNorm(div_sqrt(in_features), in_features)
        elif "instance" in normalization.lower():
            self.normalization = nn.InstanceNorm2d(in_features)
        else:
            raise NotImplementedError(f"normalization={normalization} not supported")

        if activation is None or "none" in activation.lower() or "null" in activation.lower():
            self.activation = nn.Identity()
        elif "leaky" in activation.lower():
            self.activation = nn.LeakyReLU(0.2)
        elif "relu" in activation.lower():
            self.activation = nn.ReLU()
        elif "selu" in activation.lower():
            self.activation = nn.SELU()
        elif "gelu" in activation.lower():
            self.activation = nn.GELU()
        else:
            raise NotImplementedError(f"activation={activation} not supported")

    def forward(self, x: Tensor) -> Tensor:
        out = self.normalization(x)
        out = self._conv_forward(out, self.weight * self.scale, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        if self.residual:
            return (out + x) / sqrt(2)
        return out


# ******************************************************************************************************************** #


class ConvBlock(nn.Sequential):
    def __init__(
            self,
            in_features: int,
            out_features: int,

            # ConvBlock params
            n_layers: int = 2,
            down_sample: Union[bool, int, nn.Module] = False,
            up_sample: Union[bool, int, nn.Module] = False,

            # ConvLayer params
            normalization: Optional[str] = "batchnorm",
            activation: Optional[str] = "relu",
            equalizer_lr: bool = False,
            dropout: float = 0.,

            # nn.Conv2d params
            kernel_size: _size_2_t = 3,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 1,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            residual: bool = False
    ) -> None:
        """
        Conv block that wraps a series of `n_layers` ConvLayer with up/down sample.
        Inspired from `StyleGAN <https://arxiv.org/abs/1812.04948>`_

        :param in_features: Number of channels in the input image
        :param out_features: Number of channels produced by the convolution

        :param n_layers: Num conv layers. Unused if parameter `features` provided.
        :param down_sample: If ``True``, perform 0.5 x bilinear interpolation before applying the ConvLayers.
                            If integer, perform 1/down_sample x bilinear interpolation before applying the ConvLayers.
                            If nn.Module simply apply before applying the ConvLayers.
                            Default ``False``
        :param up_sample: If ``True``, perform 2 x bilinear interpolation after applying the ConvLayers.
                          If integer, perform up_sample x bilinear interpolation after applying the ConvLayers.
                          If nn.Module simply apply after applying the ConvLayers.
                          Default ``False``

        :param residual: If ``True``, add the input to the output tensor (skip connection). Default: ``False``
        :param normalization: Normalization layer. nn.Module expecting parameter ``num_features`` in `__init__`.
                              Default ``nn.BatchNorm2d``
        :param activation: Activation layer. nn.Module expecting no parameter in ``__init__``. Default ``nn.ReLU``
        :param equalizer_lr: If ``True`` normalize the convolution weights by (in_features * kernel_size ** 2).
                             Default ``False``
        :param dropout: Dropout probability

        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param padding: Padding added to all four sides of the input. Default: 0
        :param dilation: Spacing between kernel elements. Default: 1
        :param groups:  Number of blocked connections from input channels to output channels. Default: 1
        :param bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        """
        if isinstance(down_sample, nn.Module): down_sample = down_sample
        elif isinstance(down_sample, int) and down_sample > 0: down_sample = nn.Upsample(scale_factor=1/down_sample)
        elif isinstance(down_sample, bool) and down_sample: down_sample = nn.Upsample(scale_factor=0.5)
        else: down_sample = nn.Identity()

        if isinstance(up_sample, nn.Module): up_sample = up_sample
        elif isinstance(up_sample, int) and up_sample > 0: up_sample = nn.Upsample(scale_factor=up_sample)
        elif isinstance(up_sample, bool) and up_sample: up_sample = nn.Upsample(scale_factor=2)
        else: up_sample = nn.Identity()

        super().__init__(
            down_sample,
            ConvLayer(in_features, out_features, residual, normalization, activation, equalizer_lr,
                      dropout, kernel_size, stride, padding, dilation, groups, bias),
            *([ConvLayer(out_features, out_features, residual, normalization, activation, equalizer_lr,
                         dropout, kernel_size, stride, padding, dilation, groups, bias)] * (n_layers - 1)),
            up_sample
        )


# ******************************************************************************************************************** #


class CNN(nn.Sequential):
    def __init__(
            self,
            # CNN params
            in_features: int,
            out_features: int,
            in_resolution: Optional[int] = None,
            out_resolution: Optional[int] = None,
            intermediate_features: Optional[List[int]] = None,
            capacity: int = 8,

            # ConvBlock params
            n_layers: int = 2,
            down_sample: Union[bool, int] = False,
            up_sample: Union[bool, int] = False,

            # ConvLayer params
            normalization: Optional[str] = "batchnorm",
            activation: Optional[str] = "relu",
            equalizer_lr: bool = False,
            dropout: float = 0.,

            # nn.Conv2d params
            kernel_size: _size_2_t = 3,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 1,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            residual: bool = False
    ) -> None:
        """
        `PyTorch <https://pytorch.org/>`_ implementation of a modular CNN

        Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

        .. warning:: Work in progress. This implementation is still being verified.

        .. _TheoA: https://github.com/theoad

        :param in_features: Number of channels in the input image
        :param out_features: Number of channels produced by the network
        :param in_resolution: Resolution of the input image. Can be left unfilled if `features` is provided
        :param out_resolution: Resolution of the image produced by the network. Can left unfilled if `features` provided
        :param intermediate_features: Optional list of features to create the network. If left unfilled,
                                      number of conv block is inferred from the input/output resolution ratio
                                      and features are doubled with each conv block.
        :param capacity: Channel after the first conv. Unused if parameter `features` provided.

        :param n_layers: Num conv layers. Unused if parameter `features` provided.
        :param down_sample: If ``True``, perform 0.5 x bilinear interpolation before applying the ConvLayers.
                            If integer, perform 1/down_sample x bilinear interpolation before applying the ConvLayers.
                            Default ``False``
        :param up_sample: If ``True``, perform 2 x bilinear interpolation after applying the ConvLayers.
                          If integer, perform up_sample x bilinear interpolation after applying the ConvLayers.
                          Default ``False``

        :param residual: If ``True``, add the input to the output tensor (skip connection). Default: ``False``
        :param normalization: Normalization layer. nn.Module expecting parameter ``num_features`` in `__init__`.
                              Default ``nn.BatchNorm2d``
        :param activation: Activation layer. nn.Module expecting no parameter in ``__init__``. Default ``nn.ReLU``
        :param equalizer_lr: If ``True`` normalize the convolution weights by (in_features * kernel_size ** 2).
                             Default ``False``
        :param dropout: Dropout probability

        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param padding: Padding added to all four sides of the input. Default: 0
        :param dilation: Spacing between kernel elements. Default: 1
        :param groups:  Number of blocked connections from input channels to output channels. Default: 1
        :param bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        """
        if bool(up_sample) and bool(down_sample):
            raise ValueError("Both `up_sample` and `down_sample` are set.")
        if intermediate_features is not None:
            features = [in_features] + intermediate_features + [out_features]
        else:
            if not all([in_resolution is not None, out_resolution is not None, bool(up_sample) or bool(down_sample)]):
                raise ValueError("`features` is None. Set `in_resolution`, `out_resolution` and"
                                 " (`up_sample` or `down_sample`)  to infer number of blocks")
            if bool(down_sample):
                if in_resolution <= out_resolution:
                    raise ValueError("`down_sample` set but `in_resolution` < `out_resolution`")
                if isinstance(down_sample, bool): down_sample = 2
                features, resolutions = get_channel_list(
                    in_features, out_features, in_resolution, out_resolution, down_sample, capacity
                )
            elif bool(up_sample):
                if out_resolution <= in_resolution:
                    raise ValueError("`up_sample` set but `out_resolution` < `in_resolution`")
                if isinstance(up_sample, bool): up_sample = 2
                features, resolutions = get_channel_list(
                    out_features, in_features, out_resolution, in_resolution, up_sample, capacity
                )
                features, resolutions = features[::-1], resolutions[::-1]
            else:
                raise ValueError("`features` is None. Set `in_resolution`, `out_resolution` and"
                                 " (`up_sample` or `down_sample`)  to infer number of blocks")

        super().__init__(
            ConvLayer(features[0], features[1], equalizer_lr=equalizer_lr, kernel_size=1, padding=0),
            *[ConvBlock(ic, oc, n_layers, down_sample, up_sample, normalization, activation, equalizer_lr,
                        dropout, kernel_size, stride, padding, dilation, groups, bias, residual)
              for ic, oc in zip(features[1:-2], features[2:-1])],
            ConvLayer(features[-2], features[-1], equalizer_lr=equalizer_lr, kernel_size=1, padding=0)
        )


# ******************************************************************************************************************** #


def get_block_scaling(max_resolution: int, min_resolution: int, max_scaling: int) -> List[int]:
    """
    Generates consecutive scaling factors to go from high resolution to low resolution

    :param max_resolution: max resolution
    :param min_resolution: min resolution
    :param max_scaling: highest scaling factor
    :return: ex: get_block_scaling(64, 2, 4) --> [4, 4, 2]
    """
    log_res_ratio = int(log2(max_resolution // min_resolution))
    log_scale = int(log2(max_scaling))
    mapping = []
    while log_res_ratio > 0:
        mapping.extend([int(2 ** log_scale)] * (log_res_ratio // log_scale))
        log_res_ratio %= log_scale
        log_scale -= 1
    return mapping


# ******************************************************************************************************************** #


def get_channel_list(
        in_features: int,
        out_features: int,
        in_resolution: int,
        out_resolution: int,
        scaling_factor: int,
        capacity: int):
    """
    Generates a list of integers representing channels of a CNN encoder
    where channels get doubled and resolution decreases.

    :param in_features: Number of channels in the input image
    :param out_features: Number of channels produced by the network
    :param in_resolution: Resolution of the input image.
    :param out_resolution: Resolution of the image produced by the network.
    :param capacity: Channel after the first conv.
    :param scaling_factor: maximal resolution decrease per layer
    :return: ex: get_channel_list(3, 256, 128, 4, 2, 16) -->
                 channels = [3, 16, 32, 64, 128, 256]
                 resolutions = [128, 64, 32, 16, 8, 4]
    """
    scaling_factors = get_block_scaling(in_resolution, out_resolution, scaling_factor)
    features = [max(min(2 ** i * capacity, out_features), in_features) for i, sf in enumerate(scaling_factors)]
    resolutions = [in_resolution]
    for sf in scaling_factors: resolutions.append(resolutions[-1] // sf)
    features[-1] = out_features
    features = [in_features] + features
    return features, resolutions


# ******************************************************************************************************************** #


def div_sqrt(n: int) -> int:
    """
    Return `n` divisor that is closest to sqrt(n).

    :param n: positive integer
    :return: divisor of n that is the closest to sqrt(n)
    """
    assert isinstance(n, int) and n > 0, f"Error, n must be a positive integer. Given n={n}."
    divs = np.array(divisors(n))
    if len(divs) < 1:
        return 1
    sqrt_idx = np.searchsorted(divs, sqrt(n))
    div = divs[sqrt_idx]
    return div

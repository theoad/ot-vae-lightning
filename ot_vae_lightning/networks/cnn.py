"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a modular CNN

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import math
import warnings
import torch
from torch import Tensor
from typing import Union, List, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from math import sqrt, log2
import numpy as np
from sympy import divisors
from ot_vae_lightning.networks.nets_utils import GaussianFourierProjection, FilterSequential, QKVAttention

__all__ = ['ConvLayer', 'ConvBlock', 'AttentionBlock', 'CNN', 'AutoEncoder']


# ******************************************************************************************************************** #


class ConvLayer(nn.Conv2d):
    """
    nn.Conv2 Layer that supports:
    -   residual skip connections
    -   normalization
    -   activation
    -   up-sampling
    -   down-sampling
    -   `equalized learning rate <https://arxiv.org/abs/1710.10196>`_

    The conv, norm, act, sample ordering is inspired from  `BigGAN <https://arxiv.org/pdf/1809.11096.pdf>`_
    """

    enable_warnings = False

    def __init__(
            self,
            in_features: int,
            out_features: int,

            # ConvLayer params
            down_sample: Union[bool, int, nn.Module] = False,
            up_sample: Union[bool, int, nn.Module] = False,
            additional_embed: Optional[int] = None,
            normalization: Optional[str] = None,
            activation: Optional[str] = None,
            equalized_lr: Optional[float] = None,
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
        :param in_features: Number of channels in the input image
        :param out_features: Number of channels produced by the convolution
        :param down_sample: If ``True``, perform 0.5 x bilinear interpolation before applying the ConvLayers.
                            If integer, perform 1/down_sample x bilinear interpolation before applying the ConvLayers.
                            If nn.Module simply apply before applying the ConvLayers.
                            Default ``False``
        :param up_sample: If ``True``, perform 2 x bilinear interpolation after applying the ConvLayers.
                          If integer, perform up_sample x bilinear interpolation after applying the ConvLayers.
                          If nn.Module simply apply after applying the ConvLayers.
                          Default ``False``

        :param bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        :param normalization: Normalization layer. nn.Module expecting parameter ``num_features`` in `__init__`.
                              Default None
        :param activation: Activation layer. nn.Module expecting no parameter in ``__init__``. Default `None
        :param equalized_lr: If ``True`` normalize the convolution weights by (in_features * kernel_size ** 2).
                             Default ``False``
        :param dropout: Dropout probability

        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param padding: Padding added to all four sides of the input. Default: 0
        :param dilation: Spacing between kernel elements. Default: 1
        :param groups:  Number of blocked connections from input channels to output channels. Default: 1
        """
        if isinstance(down_sample, nn.Module): down_sample = down_sample
        else:
            if bool(down_sample):
                kernel_size = max(2 * int(down_sample), kernel_size)
                stride = 2 if isinstance(down_sample, bool) else int(down_sample)
                padding = (kernel_size - 1) // 2
            down_sample = nn.Identity()
        groups = groups if in_features % groups == 0 and out_features % groups == 0 else 1
        super().__init__(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias)
        self._down_sample = down_sample

        if isinstance(up_sample, nn.Module): up_sample = up_sample
        elif isinstance(up_sample, int) and up_sample > 0: up_sample = nn.Upsample(scale_factor=up_sample)
        elif isinstance(up_sample, bool) and up_sample: up_sample = nn.Upsample(scale_factor=2)
        else: up_sample = nn.Identity()
        self._up_sample = up_sample

        self._dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self._lr_mult, self._gain = equalized_lr or 1, 1
        self._embed_proj_scale = nn.Linear(additional_embed, in_features) if bool(additional_embed) else None
        self._embed_proj_bias = nn.Linear(additional_embed, in_features) if bool(additional_embed) else None
        self._conv_scale = self._gain / np.sqrt(np.prod(self.weight.shape[1:])) if equalized_lr else 1
        self._linear_scale = self._gain / np.sqrt(in_features) if equalized_lr else 1

        if normalization is None or "none" in normalization.lower() or "null" in normalization.lower():
            self._normalization = nn.Identity()
        elif "batch" in normalization.lower(): self._normalization = nn.BatchNorm2d(in_features)
        elif "group" in normalization.lower(): self._normalization = nn.GroupNorm(div_sqrt(in_features // groups), in_features)
        elif "instance" in normalization.lower(): self._normalization = nn.InstanceNorm2d(in_features)
        else: raise NotImplementedError(f"normalization={normalization} not supported")

        if activation is None or "none" in activation.lower() or "null" in activation.lower():
            self._activation = nn.Identity()
        elif "leaky" in activation.lower():
            self._activation = nn.LeakyReLU(0.2)
            torch.nn.init.kaiming_uniform_(self.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        elif "relu" in activation.lower():
            self._activation = nn.ReLU()
            torch.nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')
        elif "selu" in activation.lower():
            self._activation = nn.SELU()
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('selu'))
        elif "gelu" in activation.lower():
            self._activation = nn.GELU()
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('linear'))
        elif "silu" in activation.lower() or "swish" in activation.lower():
            self._activation = nn.SiLU()
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('linear'))
        else: raise NotImplementedError(f"activation={activation} not supported")

        if equalized_lr:
            nn.init.normal_(self.weight, std=1/self._lr_mult)
            # if self.bias is not None: nn.init.zeros_(self.bias)
            if self._embed_proj_scale:
                nn.init.normal_(self._embed_proj_scale.weight, std=1/self._lr_mult)
                # nn.init.zeros_(self._embed_proj_scale.bias)
            if self._embed_proj_bias:
                nn.init.normal_(self._embed_proj_bias.weight, std=1/self._lr_mult)
                # nn.init.zeros_(self._embed_proj_bias.bias)

    def _add_embed(self, x: Tensor, embed: Tensor) -> Tensor:
        if self.enable_warnings:
            if embed is not None and self._embed_proj_scale is None:
                warnings.warn("""
                given conditional argument `embed` but `self.embed_proj` is None.
                To enable embedding-conditioned layer, use `layer = ConvLayer(additional_embed=...).`
                """)
            if self._embed_proj_scale is not None and embed is None:
                warnings.warn("""
                `additional_embed` specified in ConvLayer constructor but `embed` is None.
                ignoring additional `embed` input.
                """)
        if self._embed_proj_scale is not None:
            assert embed is not None, "Flow assertion error. Expected `embed` to not be None"
            scale = F.linear(
                self._activation(embed),
                self._embed_proj_scale.weight * self._linear_scale * self._lr_mult,
                self._embed_proj_scale.bias * self._lr_mult
            )[..., None, None]
            bias = F.linear(
                self._activation(embed),
                self._embed_proj_bias.weight * self._linear_scale * self._lr_mult,
                self._embed_proj_bias.bias * self._lr_mult
            )[..., None, None]
            x = x * scale + bias
        return x

    def forward(self, x: Tensor, embed: Optional[Tensor] = None) -> Tensor:
        out = self._normalization(x)
        out = self._add_embed(out, embed)
        out = self._activation(out)
        out = self._up_sample(out)
        out = self._conv_forward(out, self.weight * self._conv_scale * self._lr_mult,
                                 self.bias * self._lr_mult if self.bias is not None else None)
        out = self._down_sample(out)
        out = self._dropout(out)
        return out


class Conv1x1(ConvLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            **kwargs  # overrides the defaults
    ):
        defaults = dict(
            down_sample=False, up_sample=False, additional_embed=None, normalization=None, activation=None, equalized_lr=False,
            dropout=0., stride=1, kernel_size=1, padding=0, dilation=1, groups=1, bias=False,
        )
        super().__init__(in_features, out_features, **{**defaults, **kwargs})


# ******************************************************************************************************************** #


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Inspired from `lucidrains <https://github.com/lucidrains/denoising-diffusion-pytorch>`_' implementation.
    """
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        additional_embed: Optional[int] = None,
        normalization: Optional[str] = None,
        equalized_lr: Optional[float] = None,
        groups: int = 1,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"q,k,v channels: {channels} is not divisible by heads: {heads}")
        self.qkv = Conv1x1(channels, channels * 3, additional_embed=additional_embed, normalization=normalization,
                           equalized_lr=equalized_lr, groups=groups)
        self.attention = QKVAttention(heads)
        self.proj_out = Conv1x1(channels, channels, equalized_lr=equalized_lr, groups=groups)

    def forward(self, x: Tensor, embed: Optional[Tensor] = None) -> Tensor:
        b, c, *spatial = x.shape
        qkv = self.qkv(x, embed=embed).flatten(2)
        h = self.attention(qkv).unflatten(2, spatial)
        h = self.proj_out(h)
        return h


# ******************************************************************************************************************** #


class ConvBlock(nn.Module):
    """
    Conv block that wraps a series of `n_layers`.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,

            # ConvBlock params
            n_attn_heads: int = 0,
            n_layers: int = 2,

            # ConvLayer params
            down_sample: Union[bool, int, nn.Module] = False,
            up_sample: Union[bool, int, nn.Module] = False,
            additional_embed: Optional[int] = None,
            normalization: Optional[str] = "batchnorm",
            activation: Optional[str] = "relu",
            residual: Optional[str] = None,
            equalized_lr: Optional[float] = None,
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
        :param equalized_lr: If ``True`` normalize the convolution weights by (in_features * kernel_size ** 2).
                             Default ``False``
        :param dropout: Dropout probability

        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param padding: Padding added to all four sides of the input. Default: 0
        :param dilation: Spacing between kernel elements. Default: 1
        :param groups:  Number of blocked connections from input channels to output channels. Default: 1
        :param bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        """
        super().__init__()
        self.out_features = out_features
        embed_features = out_features // 2 if residual == "cat" else out_features
        self.block = FilterSequential(
            # Down/Up-sampling Conv
            ConvLayer(in_features, embed_features, down_sample, up_sample, additional_embed, normalization, activation,
                      equalized_lr, dropout, kernel_size, stride, padding, dilation, groups, bias),

            # Fixed spatial extent Conv
            *[ConvLayer(embed_features, embed_features, False, False, additional_embed, normalization, activation,
                        equalized_lr, dropout, kernel_size, stride, padding, dilation, groups, bias)
              for _ in range((n_layers - 1))],

            # Attention
            AttentionBlock(embed_features, n_attn_heads, additional_embed, normalization, equalized_lr, groups)
            if n_attn_heads > 0 else nn.Identity(),
        )

        # Residual (skip) connection
        self.residual = residual
        self.skip = Conv1x1(
            in_features, embed_features, down_sample=down_sample, up_sample=up_sample,
            normalization=normalization, equalized_lr=equalized_lr, groups=groups
        ) if residual in ["cat", "add"] else None

    def forward(self, x: Tensor, embed: Optional[Tensor] = None) -> Tensor:
        out = self.block(x, embed=embed)
        if self.residual == "cat": return torch.cat([out, self.skip(x)], dim=1)
        elif self.residual == "add": return out + self.skip(x)
        else: return out


# ******************************************************************************************************************** #


class CNN(FilterSequential):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a modular CNN

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(
            self,
            # CNN params
            in_features: int,
            out_features: int,
            in_resolution: Optional[int] = None,
            out_resolution: Optional[int] = None,
            intermediate_features: Optional[List[int]] = None,
            capacity: int = 8,
            max_attn_res: int = 16,

            # ConvBlock params
            n_layers: int = 2,
            residual: Optional[str] = None,
            down_sample: Union[nn.Module, bool, int] = False,
            up_sample: Union[nn.Module, bool, int] = False,

            # ConvLayer params
            additional_embed: Optional[int] = None,
            normalization: Optional[str] = "batchnorm",
            activation: Optional[str] = "relu",
            equalized_lr: Optional[float] = None,
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
        :param in_features: Number of channels in the input image
        :param out_features: Number of channels produced by the network
        :param in_resolution: Resolution of the input image. Can be left unfilled if `features` is provided
        :param out_resolution: Resolution of the image produced by the network. Can be left unfilled if `features` provided
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
        :param equalized_lr: If ``True`` normalize the convolution weights by (in_features * kernel_size ** 2).
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

            # setting this such that we don't use attention since we don't know the actual spatial extent.
            attn_resolutions = [max_attn_res] * len(features)
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
                attn_resolutions = resolutions[1:]
            elif bool(up_sample):
                if out_resolution <= in_resolution:
                    raise ValueError("`up_sample` set but `out_resolution` < `in_resolution`")
                if isinstance(up_sample, bool): up_sample = 2
                features, resolutions = get_channel_list(
                    out_features, in_features, out_resolution, in_resolution, up_sample, capacity
                )
                features, resolutions = features[::-1], resolutions[::-1]
                attn_resolutions = resolutions[:-1]
            else: raise NotImplementedError("Should never get here")
        heads = lambda ch, res: div_sqrt(ch) if res <= max_attn_res else 0
        super().__init__(
            # Conv1x1(features[0], features[0], normalization=normalization, additional_embed=additional_embed,
            #         equalized_lr=equalized_lr, groups=groups),
            *[ConvBlock(ic, oc, heads(oc, r), n_layers, down_sample, up_sample, additional_embed, normalization,
                        activation, residual, equalized_lr, dropout, kernel_size, stride, padding, dilation, groups,
                        bias) for ic, oc, r in zip(features[:-1], features[1:], attn_resolutions)],
            # Conv1x1(features[-1], features[-1], normalization=normalization, additional_embed=additional_embed,
            #         equalized_lr=equalized_lr, groups=groups),
        )
        self.out_size = torch.Size([out_features, out_resolution, out_resolution])

    def forward(self, x: Tensor, embed: Optional[Tensor] = None) -> Tensor:
        return super().forward(x, embed=embed)

# ******************************************************************************************************************** #


class AutoEncoder(nn.Module):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a modular CNN

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(
            self,
            # CNN params
            in_features: int,
            latent_features: int,
            in_resolution: Optional[int] = None,
            latent_resolution: Optional[int] = None,
            intermediate_features: Optional[List[int]] = None,
            capacity: int = 8,
            max_attn_res: int = 16,
            num_classes: Optional[int] = None,
            time_embed_dim: Optional[int] = None,
            double_encoded_features: bool = False,  # for re-parametrization trick

            # ConvBlock params
            n_layers: int = 2,
            residual: Optional[str] = None,
            down_up_sample: Union[bool, int] = False,

            # ConvLayer params
            normalization: Optional[str] = "batchnorm",
            activation: Optional[str] = "relu",
            equalized_lr: Optional[float] = None,
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
        :param in_features: Number of channels in the input image
        :param latent_features: Number of channels in the latent space
        :param double_encoded_features: if ``True`` will double the number of out_features of the encoder
                                        (for re-parametrization trick)
        :param in_resolution: Resolution of the input image. Can be left unfilled if `features` is provided
        :param latent_resolution: Resolution of the latent maps. Can left unfilled if `features` provided
        :param intermediate_features: Optional list of features to create the encoder. If left unfilled,
                                      number of conv block is inferred from the input/output resolution ratio
                                      and features are doubled with each conv block.
        :param capacity: Channel after the first conv. Unused if parameter `features` provided.

        :param n_layers: Num conv layers. Unused if parameter `features` provided.
        :param down_up_sample: If ``True``, perform 0.5 x bilinear interpolation before/after applying the ConvLayers.
                               If integer, perform 1/down_up_sample x bilinear interpolation before applying the
                               ConvLayers in the encoder and down_up_sample x bilinear after appying the ConvLayers in
                               the decoder. Default ``False``

        :param residual: If ``True``, add the input to the output tensor (skip connection). Default: ``False``
        :param normalization: Normalization layer. nn.Module expecting parameter ``num_features`` in `__init__`.
                              Default ``nn.BatchNorm2d``
        :param activation: Activation layer. nn.Module expecting no parameter in ``__init__``. Default ``nn.ReLU``
        :param equalized_lr: If ``True`` normalize the convolution weights by (in_features * kernel_size ** 2).
                             Default ``False``
        :param dropout: Dropout probability

        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param padding: Padding added to all four sides of the input. Default: 0
        :param dilation: Spacing between kernel elements. Default: 1
        :param groups:  Number of blocked connections from input channels to output channels. Default: 1
        :param bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        """
        super().__init__()
        cls_embed = 2 ** (int(math.log10(num_classes)) + 5) if bool(num_classes) else None
        self.latent_size = torch.Size([latent_features * (1 + int(double_encoded_features)), latent_resolution, latent_resolution])
        self.class_embed = nn.Embedding(num_classes, cls_embed) if bool(num_classes) else None
        self.time_embed = GaussianFourierProjection(time_embed_dim, time_embed_dim) if bool(time_embed_dim) else None

        if bool(self.class_embed) and bool(self.time_embed): additional_embed = cls_embed + time_embed_dim
        elif bool(self.class_embed): additional_embed = cls_embed
        elif bool(self.time_embed): additional_embed = time_embed_dim
        else: additional_embed = None

        self.encoder = CNN(
            in_features, latent_features * (1 + int(double_encoded_features)), in_resolution, latent_resolution,
            intermediate_features, capacity, max_attn_res, n_layers, residual, down_up_sample, False, additional_embed,
            normalization, activation, equalized_lr, dropout, kernel_size, stride, padding, dilation, groups, bias
        )

        self.decoder = CNN(
            latent_features, in_features, latent_resolution, in_resolution,
            intermediate_features[::-1] if intermediate_features is not None else None, capacity, max_attn_res,
            n_layers, residual, False, down_up_sample, additional_embed, normalization, activation, equalized_lr,
            dropout, kernel_size, stride, padding, dilation, groups, bias
        )

    def embed(self, labels: Optional[Tensor] = None, time: Optional[Tensor] = None):
        class_embed, time_embed = None, None
        if labels is not None and self.class_embed is None:
            warnings.warn("""
            given conditional argument `labels` but `self.class_embed` is None.
            To enable class-conditioned CNNs, use `ae = AutoEncoder(num_classes=...).`
            """)

        if self.class_embed is not None and labels is None:
            raise ValueError("`num_classes` specified but `labels` is None. Can't infer the class embedding.")
        if self.class_embed is not None:
            assert labels is not None, "Flow assertion error. Expected `labels` to not be None"
            class_embed = self.class_embed(labels)

        if time is not None and self.time_embed is None:
            warnings.warn("""
            given conditional argument `time` but `self.time_embed` is None.
            To enable time-conditioned CNNs, use `ae = AutoEncoder(time_dependant=True).`
            """)
        if self.time_embed is not None and time is None:
            raise ValueError("`time_dependant` specified but `time` is None. Can't infer the time embedding.")
        if self.time_embed is not None:
            assert time is not None, "Flow assertion error. Expected `time` to not be None"
            time_embed = self.time_embed(time)

        if class_embed is not None and time_embed is not None: return torch.cat([class_embed, time_embed], dim=1)
        elif class_embed is not None and time_embed is None: return class_embed
        elif class_embed is None and time_embed is not None: return time_embed
        else: return None

    def encode(self, x: Tensor, labels: Optional[Tensor] = None, time: Optional[Tensor] = None) -> Tensor:
        return self.encoder(x, self.embed(labels, time))

    def decode(self, z: Tensor, labels: Optional[Tensor] = None, time: Optional[Tensor] = None) -> Tensor:
        return self.decoder(z, self.embed(labels, time))

    def forward(self, x: Tensor, labels: Optional[Tensor] = None, time: Optional[Tensor] = None) -> Tensor:
        return self.decode(self.encode(x, labels, time), labels, time)

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
    if len(divs) < 1: return 1
    sqrt_idx = np.searchsorted(divs, sqrt(n))
    div = divs[sqrt_idx]
    return div

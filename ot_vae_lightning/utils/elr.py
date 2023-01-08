"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of `Equalized Learning Rate <https://arxiv.org/abs/1710.10196>`_

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Optional
import functools
import numpy as np

import torch.nn as nn
import torch.nn.utils.parametrize as P
from torch.types import _size

__all__ = ['EqualizedLR', 'equalized_lr', 'apply_equalized_lr']


class EqualizedLR(nn.Module):
    """
    Applies weight rescaling by the squared-root inverse of the weight fan-in at run-time.

    .. Example:

    >>> import torch.nn.utils.parametrize as P
    >>> elr = EqualizedLR(gain=1., lr_multiplier=0.2)
    >>> m = nn.Linear(16, 32)
    >>> P.register_parametrization(m, 'weight', elr)
    """
    def __init__(
            self,
            fan_in_dims: Optional[_size] = None,
            gain: float = 1.,
            lr_multiplier: float = 1.,
            is_bias: bool = False
    ):
        r"""
        :param gain: also known as the \alpha parameter
        :param lr_multiplier: also known as the \beta parameter
        :param is_bias: if ``True`` will only multiply the weight by the `lr_multiplier`
        """
        super().__init__()
        self.gain = gain
        self.lr_multiplier = lr_multiplier
        self.is_bias = is_bias
        self.fan_in_dims = fan_in_dims

    def forward(self, weight: nn.Parameter) -> nn.Parameter:
        if self.is_bias: return weight * self.lr_multiplier
        fan_in = np.prod([weight.shape[d] for d in self.fan_in_dims])
        return weight * self.lr_multiplier * self.gain / fan_in ** 0.5


def equalized_lr(m: nn.Module, gain: float = 1., lr_multiplier: float = 1., init_weights: bool = True) -> None:
    if type(m) != nn.Conv2d and type(m) != nn.Linear and type(m) != nn.ConvTranspose2d: return
    if init_weights:
        nn.init.normal_(m.weight, std=1/lr_multiplier)
        nn.init.zeros_(m.bias)
    fan_in_dims = [1] if type(m) == nn.Linear else [1,2,3]
    P.register_parametrization(m, 'weight', EqualizedLR(fan_in_dims, gain, lr_multiplier, is_bias=False))
    P.register_parametrization(m, 'bias', EqualizedLR([0], lr_multiplier, is_bias=True))


def apply_equalized_lr(net, gain: float = 1., lr_multiplier: float = 1., init_weights: bool = True) -> None:
    """
    Reparametrizes every nn.Conv2d, nn.ConvTransposed2d and nn.Linear to support Equalized Learning Rate.
    >>> from torchvision.models import resnet50, ResNet50_Weights
    >>> net = resnet50(ResNet50_Weights)
    >>> apply_equalized_lr(net, gain=1., lr_multiplier=1., init_weights=False)

    :param net: The network to modify
    :param gain: the learning rate gain
    :param lr_multiplier: the learning rate multiplier
    :param init_weights: if ``True`` will reinitialize all weights with a normal distribution and all biases to 0.
    """
    from torchvision.models import resnet50, ResNet50_Weights
    net.apply(functools.partial(equalized_lr, gain=gain, lr_multiplier=lr_multiplier, init_weights=init_weights))

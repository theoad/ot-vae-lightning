"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of an abstract Prior Module

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from math import cos, pi

import torch
from torch import Tensor
from torch.types import _size
import torch.nn as nn
from torch.distributions import Distribution


class Prior(nn.Module, ABC):
    """
    Prior abstract class.
    """
    def __init__(self, loss_coeff: float = 1., annealing_steps: int = 0):
        """
        :param loss_coeff: balancing coefficient of the prior loss
        """
        super().__init__()
        self._loss_coeff = loss_coeff
        self.annealing_steps = annealing_steps

    @abstractmethod
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def sample(self, shape, device) -> Tensor:
        pass

    @abstractmethod
    def out_size(self, size: _size) -> _size:
        pass

    @staticmethod
    def empirical_reverse_kl(p: Distribution, q: Distribution, z: Optional[Tensor] = None) -> Tensor:
        return q.log_prob(z) - p.log_prob(z)

    @property
    def loss_coeff(self):
        return self._loss_coeff

    def forward(self, x: Tensor, step: int) -> Tuple[Tensor, Tensor]:
        z, loss = self.encode(x)
        coeff = self.loss_coeff * (0.5 * cos(pi * (step / self.annealing_steps + 1)) + 0.5)\
            if self.annealing_steps > step else self.loss_coeff
        return z, loss * coeff

"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of an abstract Prior Module

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""

from typing import Tuple
from abc import ABC, abstractmethod
from torch import Tensor
import torch.nn as nn
from torch.distributions import Distribution


class Prior(nn.Module, ABC):
    """
    Prior abstract class.
    """
    def __init__(self, loss_coeff: float = 1.):
        """
        :param loss_coeff: balancing coefficient of the prior loss
        """
        super().__init__()
        self._loss_coeff = loss_coeff

    @abstractmethod
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def sample(self, shape, device) -> Tensor:
        pass

    @staticmethod
    def empirical_reverse_kl(
            p: Distribution,
            q: Distribution,
            z: Tensor=None
    ) -> Tensor:
        return q.log_prob(z) - p.log_prob(z)

    @property
    def loss_coeff(self):
        return self._loss_coeff

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, loss = self.encode(x)
        return z, loss * self.loss_coeff

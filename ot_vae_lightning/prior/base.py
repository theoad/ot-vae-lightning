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

from torch import Tensor
from torch.types import _size
import torch.nn as nn
from torch.distributions import Distribution

__all__ = ['Prior']


class Prior(nn.Module, ABC):
    """
    Prior abstract class.
    """
    def __init__(self, loss_coeff: float = 1., annealing_steps: int = 0):
        """
        :param loss_coeff: balancing coefficient of the prior loss
        :param annealing_steps: the number of cosine annealing steps given to the prior loss to warm-up.
        """
        super().__init__()
        self._loss_coeff = loss_coeff
        self.annealing_steps = annealing_steps

    @abstractmethod
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Implement the encoding step.
        The method is called by `forward` and should implement the prior-related logic:
            - re-parametrization trick
            - loss computation
            - re-sampling
        """

    @abstractmethod
    def sample(self, shape, device) -> Tensor:
        """
        Implement the sampling step.
        """

    @abstractmethod
    def out_size(self, size: _size) -> _size:
        """
        Receives the size of the Tensor passed to `forward` and returns the actual size
        after the `encode` step (for example, the re-parametrization trick halves the number of channels).
        """

    @staticmethod
    def empirical_reverse_kl(p: Distribution, q: Distribution, z: Optional[Tensor] = None) -> Tensor:
        reduce_dim = list(range(1, len(z.shape)))
        return (q.log_prob(z) - p.log_prob(z)).sum(reduce_dim)

    @property
    def loss_coeff(self):
        return self._loss_coeff

    def forward(self, x: Tensor, step: int, **kwargs) -> Tuple[Tensor, Tensor]:
        annealing = (0.5 * cos(pi * (step / self.annealing_steps + 1)) + 0.5) if self.annealing_steps > step else 1
        z, loss = self.encode(x, **kwargs)   # type: ignore[arg-type]
        if isinstance(loss, dict): loss['loss'] *= self.loss_coeff * annealing
        else: loss *= self.loss_coeff * annealing
        return z, loss

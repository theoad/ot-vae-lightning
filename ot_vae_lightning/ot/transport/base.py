"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of optimal transport abstract base module

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D
from torch.types import _size, _dtype

__all__ = ['TransportOperator']


class TransportOperator(nn.Module, ABC):
    def __init__(self, dim: int, batch_shape: _size, dtype: _dtype = torch.double):
        nn.Module.__init__(self)
        self.dim = dim
        self.batch_shape = batch_shape
        self.dtype = dtype

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal states.
        """

    @property
    @abstractmethod
    def source_distribution(self) -> D.Distribution:
        """
        Returns a torch.distribution.Distribution instance
        """

    @property
    @abstractmethod
    def target_distribution(self) -> D.Distribution:
        """
        Returns a torch.distribution.Distribution instance
        """

    @abstractmethod
    def _update(self, samples: Tensor, dist: Literal['source', 'target']):
        """
        Update internal states of distribution `dist`

        :param samples: samples from the distribution `dist`. [*batch_shape, B, dim]
        :param dist: 'source' of 'target'
        """

    def update(self, source_samples: Optional[Tensor] = None, target_samples: Optional[Tensor] = None) -> None:
        """
        Update internal states.

        :param source_samples: samples from the distribution to be transported. [*batch_shape, B, dim]
        :param target_samples: samples from the target distribution to which transport.  [*batch_shape, B, dim]
        """
        if source_samples is not None: self._update(source_samples, 'source')
        if target_samples is not None: self._update(target_samples, 'target')

    @abstractmethod
    def compute(self) -> Tensor:
        """
        Compute the transport operators using the internal states.

        :return: The distance between the source and the target.
        """

    @abstractmethod
    def transport(self, inputs: Tensor) -> Tensor:
        """
        Transport the given samples to the target distribution.

        :param inputs: samples drawn from the source distribution to be transported. [*batch_shape, B, dim]
        :return: the transported samples. [*batch_shape, B, dim]
        """

    def forward(self, inputs: Tensor) -> Tensor:
        return self.transport(inputs)

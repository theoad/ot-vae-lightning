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

import torch.nn as nn
from torch import Tensor
from torch.types import _size, _dtype, _device

__all__ = ['TransportOperator']


class TransportOperator(nn.Module, ABC):
    INTERNAL_STATES = []
    OPERATORS = []

    def reset(self) -> None:
        """
        Reset internal states.
        """
        for buffer in self.INTERNAL_STATES + self.OPERATORS: getattr(self, buffer).zero_()

    @abstractmethod
    def update(self, source_samples: Optional[Tensor] = None, target_samples: Optional[Tensor] = None) -> None:
        """
        Update internal states.

        :param source_samples: samples from the distribution to be transported.
        :param target_samples: samples from the target distribution to which transport.
        """

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

        :param inputs: samples drawn from the source distribution to be transported.
        :return: the transported samples.
        """

    @abstractmethod
    def sample(
            self,
            shape: _size,
            dtype: _dtype,
            device: _device,
            from_dist: Literal['source', 'target'] = 'source'
    ) -> Tensor:
        """
        Samples from the source/target distribution using the internal statistics accumulated in `update`.

        :param shape: The shape of the tensor to be sampled.
        :param dtype: The type of the tensor to be sampled.
        :param device: The device to which the sampled tensor will be allocated.
        :param from_dist: The name of the distribution from which to sample {'source', 'target'}
        :return: A tensor sampled from the prompted distribution.
        """

    def forward(self, inputs: Tensor) -> Tensor:
        return self.transport(inputs)

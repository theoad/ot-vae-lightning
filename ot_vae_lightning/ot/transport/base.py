"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of optimal transport abstract base module

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D
import ot_vae_lightning.utils as utils

from ot_vae_lightning.ot.distribution_models.base import DistributionModel

__all__ = ['TransportOperator']


class TransportOperator(nn.Module, utils.DDPMixin, ABC):
    """
    Usage:

    tr_op = TransportOperator(2, 10)  # defines a 2 different operators

    for x1, x2, y1, y2 in zip(source1, source2, target1, target2):
        x = torch.stack([x1,x2])
        y = torch.stack([y1, y2])
        assert x.shape == y.shape == torch.Size([2, 10])
        tr_op.update(source_samples=x, target_samples=y)

    dist = tr_op.compute()
    assert dist.shape == torch.Size([2])

    samples_source = tr_op.source_distribution.sample((10,))
    samples_target = tr_op.target_distribution.sample((10,))

    new_s = next(zip(source1, source2))
    tr_op.target_distribution.prob(new_s)  # very low
    new_s_transported = tr_op(new_s)
    tr_op.target_distribution.prob(new_s_transported)  # much higher

    tr_op.reset()
    """
    def __init__(
            self,
            *size: int,
            source_model: DistributionModel,
            target_model: DistributionModel,
            reset_source: bool = True,
            reset_target: bool = True,
            store_source: bool = False,
            store_target: bool = False,
            **ddp_kwargs
    ):
        nn.Module.__init__(self)
        utils.DDPMixin.__init__(self, **ddp_kwargs)
        self.dim = size[-1]
        self.leading_shape = size[:-1]

        self.source_model = source_model
        self.target_model = target_model

        self.reset_source = reset_source
        self.reset_target = reset_target

        self.store_source = store_source
        self.store_target = store_target

        if self.store_source:
            self.register_buffer("_source_samples", None)
        if self.store_target:
            self.register_buffer("_target_samples", None)

        if self.store_source or self.store_target:
            self.warn(f"""
            The transport operator `{self.__class__.__name__}` will save all extracted features in buffers.
            For large datasets this may lead to a large memory footprint.
            """)

    def reset(self) -> None:
        if self.reset_source:
            if self.store_source:
                self._source_samples = None  # noqa
            self.source_model.reset()
        if self.reset_target:
            if self.store_target:
                self._target_samples = None  # noqa
            self.target_model.reset()

    @property
    def source_distribution(self) -> D.Distribution:
        return self.source_model.distribution

    @property
    def target_distribution(self) -> D.Distribution:
        return self.target_model.distribution

    def update(self, source_samples: Optional[Tensor] = None, target_samples: Optional[Tensor] = None) -> None:
        """
        Update internal states.

        :param source_samples: samples from the distribution to be transported. [*leading_shape, B, dim]
        :param target_samples: samples from the target distribution to which transport.  [*leading_shape, B, dim]
        """
        if source_samples is not None:
            self.source_model.update(source_samples)
            if self.store_source:
                if self._source_samples is None: self._source_samples = source_samples
                else:
                    self._source_samples = torch.cat([  # noqa
                        self._source_samples,
                        source_samples.detach().type_as(self._source_samples).requires_grad_(False)
                    ], dim=-2)

        if target_samples is not None:
            self.target_model.update(target_samples)
            if self.store_target:
                if self._target_samples is None: self._target_samples = target_samples
                else:
                    self._target_samples = torch.cat([  # noqa
                        self._target_samples,
                        target_samples.detach().type_as(self._target_samples).requires_grad_(False)
                    ], dim=-2)

    def fit_models(self):
        """
        Calls `fit` on the source and target model distribution prior to computing the transport
        """
        source_samples, target_samples = None, None
        if self.store_source:
            if self.gather:
                self._source_samples = torch.cat(self.gather(self._source_samples), dim=-2)  # noqa
            source_samples = self._source_samples
        if self.store_target:
            if self.gather:
                self._target_samples = torch.cat(self.gather(self._target_samples), dim=-2)  # noqa
            target_samples = self._target_samples

        self.source_model.fit(source_samples)
        self.target_model.fit(target_samples)

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

        :param inputs: samples drawn from the source distribution to be transported. [*leading_shape, B, dim]
        :return: the transported samples. [*leading_shape, B, dim]
        """

    def forward(self, inputs: Tensor) -> Tensor:
        return self.transport(inputs)

    def extra_repr(self) -> str:
        return f"""leading_dim={tuple(self.leading_shape)}, dim={self.dim}, reset_source={self.reset_source},
reset_target={self.reset_target}, store_source={self.store_source}, store_target={self.store_target}"""

"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of Wasserstein 2 optimal discrete transport

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Literal

import torch
from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F

from ot_vae_lightning.ot.distribution_models.codebook_model import CodebookModel
from ot_vae_lightning.ot.transport.base import TransportOperator
from ot_vae_lightning.ot.w2_utils import sinkhorn_log

__all__ = ['DiscreteTransport']


class DiscreteTransport(TransportOperator):
    def __init__(self,
                 *size: int,
                 source_cfg={},
                 target_cfg={},
                 transport_type: Literal['sample', 'argmax', 'mean'],
                 sinkhorn_reg: float = 1e-5,
                 sinkhorn_max_iter: int = 1000,
                 sinkhorn_threshold: float = 1e-6,
                 **kwargs
                 ):
        super().__init__(
            *size,
            source_model=CodebookModel(*size, **source_cfg),
            target_model=CodebookModel(*size, **target_cfg),
            **kwargs
        )
        self.transport_type = transport_type
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.sinkhorn_threshold = sinkhorn_threshold
        self.transport_matrix = None

    def reset(self) -> None:
        super().reset()
        self.transport_matrix = None

    def compute(self) -> Tensor:
        self.fit_models()

        cost = self.source_model.energy(self.target_model.codebook)
        self.transport_matrix = sinkhorn_log(
            self.source_distribution.probs,
            self.target_distribution.probs,
            cost,
            reg=self.sinkhorn_reg,
            max_iter=self.sinkhorn_max_iter,
            threshold=self.sinkhorn_threshold
        )  # [*, n, k]
        total_cost = torch.sum(cost * self.transport_matrix, dim=(-2, -1))
        return total_cost

    def transport(self, inputs: Tensor) -> Tensor:
        # assign each input to a cluster w.r.t its likelihood
        training = self.training
        self.eval()
        assignments, _, _ = self.source_model.assign(inputs)

        # get the transport plan relative to each input
        target_assignments = assignments @ self.transport_matrix

        if self.transport_type == 'mean':
            pass
        elif self.transport_type == 'argmax':
            # With this transport type, the inputs are transported to the closest component
            idx = target_assignments.argmax(-1)
            target_assignments = F.one_hot(idx, target_assignments.size(-1)).type_as(target_assignments)
        elif self.transport_type == 'sample':
            # With this transport type, the inputs are transported to a component sampled
            # according to assignment probabilities
            idx = Categorical(target_assignments).sample()
            target_assignments = F.one_hot(idx, target_assignments.size(-1)).type_as(target_assignments)
        else: raise NotImplementedError()

        # finally we the inputs are replaced with the selected target components
        transported = (target_assignments @ self.target_model.codebook).type_as(inputs)
        self.train(training)
        return transported

    def extra_repr(self) -> str:
        return super().extra_repr() + f'sinkhorn_reg={self.sinkhorn_reg}, sinkhorn_max_iter={self.sinkhorn_max_iter},' \
                                      f' sinkhorn_threshold={self.sinkhorn_threshold}'

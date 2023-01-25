"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of Wasserstein 2 optimal gmm transport

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

from ot_vae_lightning.ot.transport.base import TransportOperator
from ot_vae_lightning.ot.w2_utils import W2Mixin
from ot_vae_lightning.ot.distribution_models.gassian_mixture_model import GaussianMixtureModel

__all__ = ['GMMTransport']


class GMMTransport(TransportOperator, W2Mixin):
    r"""
    Computes an upper bound[2] on the entropy-regularized squared W2 distance[1] between the following gaussian mixtures

    .. math::

        GMM_{s}=\sum_{i} w_{bi}^{s} \mathcal{N}(\mu_{bi}^{s} , \sigma_{bi}^{s}^{2})

        GMM_{t}=\sum_{i} w_{bi}^{t} \mathcal{N}(\mu_{bi}^{t} , \sigma_{bi}^{t}^{2})

    Inspired from [2]. Optimal transport plan between the means is implemented using [1].

    [1] Marco Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transport
    [2] Yongxin Chen, Tryphon T. Georgiou and Allen Tannenbaum Optimal transport for Gaussian mixture models
    """

    def __init__(
            self, *size,
            transport_type: Literal['sample', 'argmax', 'barycenter'],
            source_cfg={},
            target_cfg={},
            transport_cfg={},
            **kwargs):
        W2Mixin.__init__(self, **transport_cfg)
        TransportOperator.__init__(
            self, *size,
            source_model=GaussianMixtureModel(*size, w2_cfg=transport_cfg, **source_cfg),
            target_model=GaussianMixtureModel(*size, w2_cfg=transport_cfg, **target_cfg),
            **kwargs
        )
        self.transport_type = transport_type
        self.transport_matrix = None

    def reset(self) -> None:
        super().reset()
        self.transport_matrix = None

    def compute(self) -> Tensor:
        self.fit_models()

        total_cost, coupling = self.batch_ot_gmm(
            self.source_model.mean,
            self.target_model.mean,
            self.source_model.variances.squeeze(),
            self.target_model.variances.squeeze(),
            weight_source=self.source_model.weights,
            weight_target=self.target_model.weights,
            max_iter=100
        )

        self.transport_matrix = coupling.type_as(self.source_model.mean)
        return total_cost

    @torch.no_grad()
    def transport(self, inputs: Tensor) -> Tensor:
        # assign each input to a cluster w.r.t its likelihood
        assignments, _, _ = self.source_model.assign(inputs.to(self.dtype))
        source_means, source_vars = self.source_model.predict_mean_var(assignments)

        # get the transport plan relative to each input
        target_assignments = assignments @ self.transport_matrix

        if self.transport_type in ['sample', 'argmax']:
            if self.transport_type == 'argmax':
                # With this transport type, the inputs are transported to the closest component
                idx = target_assignments.argmax(-1)
            elif self.transport_type == 'sample':
                # With this transport type, the inputs are transported to a component sampled
                # according to assignment probabilities
                idx = Categorical(target_assignments / target_assignments.sum(-1, keepdim=True)).sample()
            else: raise NotImplementedError()

            target_assignments = F.one_hot(idx, target_assignments.size(-1)).type_as(target_assignments)
            target_means, target_vars = self.target_model.predict_mean_var(target_assignments)

        elif self.transport_type == 'barycenter':
            # In this transport type, the inputs are transported to a smooth interpolation of all
            # the components, weighted by the assignment probabilities
            target_means, target_vars = self.gaussian_barycenter(
                self.target_model.batched_distribution.component_distribution.mean,  # noqa
                self.target_model.batched_variances,
                target_assignments,
                n_iter=100,
            )
        else:
            raise NotImplementedError()

        # finally we transport the inputs to the selected target component
        transported = self.apply_transport(
            inputs, source_means, target_means,
            *self.compute_transport_operators(source_vars, target_vars)
        ).type_as(inputs)

        return transported

    def extra_repr(self) -> str:
        return super().extra_repr() + W2Mixin.__repr__(self) + f', transport_type={self.transport_type}'

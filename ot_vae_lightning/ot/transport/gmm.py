"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of Wasserstein 2 optimal gmm transport

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import numpy as np
from typing import Optional, Literal, Union

from pytorch_lightning.utilities.distributed import gather_all_tensors, distributed_available
import torch
from torch import Tensor
from torch.distributions import Categorical, MixtureSameFamily
from torch.types import _size, _device, _dtype  # noqa
import torch.nn.functional as F

from pytorch_lightning.utilities import rank_zero_warn
from ot_vae_lightning.ot.transport.base import TransportOperator
from ot_vae_lightning.ot.w2_utils import *
from ot_vae_lightning.ot.gassian_mixture_model import GaussianMixtureModel

__all__ = ['GMMTransport']


class GMMTransport(TransportOperator):
    r"""
    Computes an upper bound[2] on the entropy-regularized squared W2 distance[1] between the following gaussian mixtures

    .. math::

        GMM_{s}=\sum_{i} w_{bi}^{s} \mathcal{N}(\mu_{bi}^{s} , \sigma_{bi}^{s}^{2})

        GMM_{t}=\sum_{i} w_{bi}^{t} \mathcal{N}(\mu_{bi}^{t} , \sigma_{bi}^{t}^{2})

    Inspired from [2]. Optimal transport plan between the means is implemented using [1].

    [1] Marco Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transport
    [2] Yongxin Chen, Tryphon T. Georgiou and Allen Tannenbaum Optimal transport for Gaussian mixture models
    """

    def __init__(self,
                 n_components_source: int,
                 n_components_target: int,
                 dim: int,
                 batch_shape: _size,
                 dtype: _dtype = torch.double,
                 diag: bool = True,
                 pg_star: float = 0.,
                 stochastic: bool = False,
                 make_pd: bool = False,
                 verbose: bool = False,
                 persistent: bool = False,
                 transport_type: Literal['sample', 'barycenter'] = 'sample',
                 smooth_assignment: bool = False
                 ):
        r"""
        :param dim: The dimensionality of the distributions to transport
        :param pg_star: Perception-distortion ratio. can be seen as temperature.
        :param stochastic: If ``True`` return (T_{s -> t}, \Sigma_w) of (19) else return (T_{s -> t}, `None`) (17)
        :param persistent: whether the buffers are part of this module's :attr:`state_dict`.
        :param make_pd: Add the minimum eigenvalue in order to make matrices pd, if needed
        """
        super().__init__(dim, batch_shape, dtype)
        self.dim = dim
        self.transport_type = transport_type
        self.pg_star = pg_star
        self.stochastic = stochastic
        self.make_pd = make_pd
        self.verbose = verbose
        self.diag = diag
        self.smooth_assignment = smooth_assignment

        self.register_buffer("_source_features", None, persistent)
        self.register_buffer("_target_features", None, persistent)
        self.gmm_source = GaussianMixtureModel(batch_shape, n_components_source, self.dim, dtype, diag=self.diag)
        self.gmm_target = GaussianMixtureModel(batch_shape, n_components_target, self.dim, dtype, diag=self.diag)
        self.transport_matrix = None

        # TODO: Update centroids using running averages (with exponential decay) to avoid storing the features
        rank_zero_warn(f"""
        The transport operator `{self.__class__.__name__}` will save all extracted features in buffers.
        For large datasets this may lead to large memory footprint.
        """)

    def reset(self) -> None:
        self._source_features = None  # noqa
        self._target_features = None  # noqa
        self.transport_matrix = None

    @property
    def source_distribution(self) -> MixtureSameFamily:
        return self.gmm_source.distribution

    @property
    def target_distribution(self) -> MixtureSameFamily:
        return self.gmm_source.distribution

    def _update(self, samples: Tensor, dist: Literal['source', 'target']):
        """
        Update internal states of distribution `dist`

        :param samples: samples from the distribution `dist`. [*batch_shape, B, dim]
        :param dist: 'source' of 'target'
        """
        if samples.shape[:-2] != self.batch_shape:
            ValueError(f"`{dist}_samples` is expected to have leading dimensions equaled to {self.batch_shape}")
        if samples.size(-1) != self.dim:
            ValueError(f"`{dist}_samples` is expected to have dimensionality equaled to {self.dim}")
        samples = samples.to(dtype=self.dtype)
        buffer = getattr(self, f"_{dist}_features")
        setattr(self, f"_{dist}_features", samples if buffer is None else torch.cat([buffer, samples], dim=-2))

    def compute(self) -> Tensor:
        if distributed_available():
            self._source_features = torch.cat(gather_all_tensors(self._source_features), dim=-2)  # noqa
            self._target_features = torch.cat(gather_all_tensors(self._target_features), dim=-2)  # noqa

        GaussianMixtureModel.fit(self.gmm_source, self._source_features, batch_size=100, lr=1e-1, n_epochs=10)
        GaussianMixtureModel.fit(self.gmm_target, self._target_features, batch_size=100, lr=1e-1, n_epochs=10)

        total_cost, coupling = batch_ot_gmm(
            self.gmm_source.means.detach().requires_grad_(False),
            self.gmm_target.means.detach().requires_grad_(False),
            self.gmm_source.variances.squeeze().detach().requires_grad_(False),
            self.gmm_target.variances.squeeze().detach().requires_grad_(False),
            self.diag,
            self.gmm_source.weights.detach().requires_grad_(False),
            self.gmm_target.weights.detach().requires_grad_(False),
            verbose=True,
            dtype=self.dtype,
            max_iter=100
        )

        self._transport_matrix = coupling  # noqa [*batch_dim, n_components_source, n_components_target]
        return total_cost

    @torch.no_grad()
    def transport(self, inputs: Tensor) -> Tensor:
        orig_dtype = inputs.dtype
        if inputs.shape[:-2] != self.batch_shape:
            ValueError(f"`inputs` is expected to have leading dimensions equaled to {self.batch_shape}")
        if inputs.size(-1) != self.dim:
            ValueError(f"`inputs` is expected to have dimensionality equaled to {self.dim}")
        inputs = inputs.to(dtype=self.dtype)

        # [*batch_dim, B, dim] --> [*batch_dim, B, comp]
        assignments = self.gmm_source.predict(inputs, smooth=self.smooth_assignment, one_hot=True)

        # [*batch_dim, B, comps] --> [*batch_dim, B, dim], [*batch_dim, B, dim, dim]
        source_means, source_vars = self.gmm_source.assign_mean_var(assignments)

        # [*batch_dim, B, n_comps] x [*batch_dim, n_coms, n_compt] --> [*batch_dim, B, n_compt]
        target_assignments_probs = (assignments @ self._transport_matrix).softmax(-1)

        if self.transport_type == 'sample':
            # [*batch_dim, B, compt]
            target_assignments = F.one_hot(Categorical(target_assignments_probs).sample(),
                                           target_assignments_probs.size(-1)).type_as(target_assignments_probs)
            # [*batch_dim, B, compt] --> [*batch_dim, B, dim], [*batch_dim, B, dim, dim]
            target_means, target_vars = self.gmm_target.assign_mean_var(target_assignments)

        elif self.transport_type == 'barycenter':
            target_means, target_vars = gaussian_barycenter(
                self.gmm_target.distribution.component_distribution.mean,        # [*batch_dim, 1, compt, dim]
                self.gmm_target.variances,   # [*batch_dim, 1, compt, dim, dim] or [*batch_dim, 1, compt, dim]
                target_assignments_probs,    # [*batch_dim, B, compt]
                diag=self.diag,
                n_iter=100,
                dtype=self.dtype
            )  # [*batch_dim, B, dim], [*batch_dim, B, dim, dim]

        else: raise NotImplementedError()

        transport_operator, cov_stochastic_noise = compute_transport_operators(
            source_vars,            # [*batch_dim, B, dim, dim] or [*batch_dim, B, dim]
            target_vars,            # [*batch_dim, B, dim, dim] or [*batch_dim, B, dim]
            stochastic=self.stochastic,
            diag=self.diag,
            pg_star=self.pg_star,
            make_pd=self.make_pd,
            verbose=self.verbose,
            dtype=self.dtype
        )  # [*batch_dim, B, dim, dim] or [*batch_dim, B, dim]

        transported = apply_transport(
            inputs,                 # [*batch_dim, B, dim]
            source_means,           # [*batch_dim, B, dim]
            target_means,           # [*batch_dim, B, dim]
            transport_operator,     # [*batch_dim, B, dim, dim] or [*batch_dim, B, dim]
            cov_stochastic_noise,   # [*batch_dim, B, dim, dim] or [*batch_dim, B, dim]
            diag=self.diag,
            dtype=self.dtype
        )  # [*batch_dim, B, dim]

        return transported.to(dtype=orig_dtype)

"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of Wasserstein 2 optimal gmm transport

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import numpy as np
from typing import Optional, Literal

from pytorch_lightning.utilities.distributed import gather_all_tensors, distributed_available
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.types import _size, _device, _dtype  # noqa

from pytorch_lightning.utilities import rank_zero_warn
from ot_vae_lightning.ot.transport.base import TransportOperator
from ot_vae_lightning.ot.w2_utils import batch_ot_gmm, gaussian_barycenter
from ot_vae_lightning.ot.transport import compute_transport_operators, apply_transport
from ot_vae_lightning.ot.gmm_distribution import GaussianMixtureModel

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
                 transport_type: Literal['sample', 'barycenter'] = 'sample',
                 pg_star: float = 0.,
                 diag: bool = True,
                 stochastic: bool = False,
                 make_pd: bool = False,
                 verbose: bool = False,
                 persistent: bool = False
                 ):
        r"""
        :param dim: The dimensionality of the distributions to transport
        :param pg_star: Perception-distortion ratio. can be seen as temperature.
        :param stochastic: If ``True`` return (T_{s -> t}, \Sigma_w) of (19) else return (T_{s -> t}, `None`) (17)
        :param persistent: whether the buffers are part of this module's :attr:`state_dict`.
        :param make_pd: Add the minimum eigenvalue in order to make matrices pd, if needed
        """
        super().__init__()
        self.dim = dim
        self.transport_type = transport_type
        self.pg_star = pg_star
        self.stochastic = stochastic
        self.make_pd = make_pd
        self.verbose = verbose
        self.diag = diag

        self.register_buffer("_source_features", None, persistent)
        self.register_buffer("_target_features", None, persistent)
        self.register_buffer("_transport_matrix", None, persistent)
        self.gmm_source = GaussianMixtureModel(n_components_source, self.dim, diag=self.diag)
        self.gmm_target = GaussianMixtureModel(n_components_target, self.dim, diag=self.diag)

        # TODO: Update centroids using running averages (with exponential decay) to avoid storing the features
        rank_zero_warn(f"""
        The transport operator `{self.__class__.__name__}` will save all extracted features in buffers.
        For large datasets this may lead to large memory footprint.
        """)

    def reset(self) -> None:
        self._source_features = None  # noqa
        self._target_features = None  # noqa
        self._transport_matrix = None  # noqa
        self.gmm_source._init_params()  # noqa
        self.gmm_target._init_params()  # noqa

    def update(self, source_samples: Optional[Tensor] = None, target_samples: Optional[Tensor] = None) -> None:
        # noinspection Duplicates
        if source_samples is not None:
            flattened = source_samples.flatten(1).double()
            if flattened.dim() != 2 or flattened.size(1) != self.dim:
                ValueError(f"`source_samples` flattened is expected to have dimensionality equaled to {self.dim}")
            if self._source_features is None:
                self._source_features = flattened
            self._source_features = torch.cat([self._source_features, flattened], dim=0)  # noqa

        # noinspection Duplicates
        if target_samples is not None:
            flattened = target_samples.flatten(1).double()
            if flattened.dim() != 2 or flattened.size(1) != self.dim:
                ValueError(f"`target_samples` flattened is expected to have dimensionality equaled to {self.dim}")
            if self._target_features is None:
                self._target_features = flattened
            self._target_features = torch.cat([self._target_features, flattened], dim=0)  # noqa

    def compute(self) -> Tensor:
        if distributed_available():
            self._source_features = torch.cat(gather_all_tensors(self._source_features), dim=0)  # noqa [N, dim]
            self._target_features = torch.cat(gather_all_tensors(self._target_features), dim=0)  # noqa [N, dim]

        # TODO: warning, this is not deterministic and will produce different results for each gpu
        self.gmm_source.fit(self._source_features)
        self.gmm_target.fit(self._target_features)

        total_cost, coupling = batch_ot_gmm(
            self.gmm_source.means,
            self.gmm_target.means,
            self.gmm_source.variances,
            self.gmm_target.variances,
            self.diag,
            self.gmm_source.weights,
            self.gmm_target.weights,
            verbose=True,
            max_iter=100
        )

        self._transport_matrix = coupling  # noqa [n_components_source, n_components_target]
        return total_cost

    def transport(self, inputs: Tensor) -> Tensor:
        flattened = inputs.flatten(1).double().to(self._transport_matrix.device)
        if flattened.dim() != 2 or flattened.size(1) != self.dim:
            ValueError(f"`inputs` flattened is expected to have dimensionality equaled to {self.dim}")

        assignments = self.gmm_source.predict(inputs)                                       # [N,]
        source_means = self.gmm_source.means[assignments]                                   # [N, dim]
        source_vars = self.gmm_source.variances[assignments]                                # [N, dim] or [N, dim, dim]
        target_assignments_probs = self._transport_matrix[assignments]                      # [N, n_components_target]

        if self.transport_type == 'sample':
            target_assignments = Categorical(target_assignments_probs.softmax(-1)).sample() # [N,]
            target_means = self.gmm_target.means[target_assignments]                        # [N, dim]
            target_vars = self.gmm_target.variances[target_assignments]                     # [N, dim] or [N, dim, dim]

        elif self.transport_type == 'barycenter':
            # [N, n_components_target] x [n_components_target, dim] --> [N, dim]
            target_means, target_vars = gaussian_barycenter(
                self.gmm_target.means,
                self.gmm_target.variances,
                target_assignments_probs,
                diag=self.diag
            )

        else: raise NotImplementedError()

        transport_operator, cov_stochastic_noise = compute_transport_operators(
            source_vars,
            target_vars,
            stochastic=self.stochastic,
            diag=self.diag,
            pg_star=self.pg_star,
            make_pd=self.make_pd,
            verbose=self.verbose
        )

        transported = apply_transport(
            flattened,
            source_means.type_as(flattened),
            target_means.type_as(flattened),
            transport_operator.type_as(flattened),
            cov_stochastic_noise.type_as(flattened),
            diag=self.diag
        )

        return transported.to(dtype=inputs.dtype).unflatten(1, inputs.shape[1:])

    def sample(
            self,
            shape: _size,
            dtype: _dtype,
            device: _device,
            from_dist: Literal['source', 'target'] = 'source'
    ) -> Tensor:
        if np.prod(shape[1:]) != self.dim:
            ValueError(f"`shape` is expected to have dimensionality equaled to {self.dim}")

        samples = getattr(self, f'gmm_{from_dist}').sample(shape[0])
        return samples

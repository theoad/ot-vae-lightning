"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of Wasserstein 2 optimal gaussian transport

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import numpy as np
from typing import Optional
from functools import partial

from pytorch_lightning.utilities.distributed import sync_ddp_if_available
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal

from ot_vae_lightning.ot.transport.base import TransportOperator
from ot_vae_lightning.ot.w2_utils import w2_gaussian
from ot_vae_lightning.ot.transport.functional import compute_transport_operators, apply_transport
from ot_vae_lightning.ot.matrix_utils import make_psd, STABILITY_CONST, mean_cov

__all__ = ['GaussianTransport']


class GaussianTransport(TransportOperator):
    r"""
    Computes the following transport operators according to eq. 17, 19 in [1]:

    .. math::

        (17) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{-0.5}_{s} (\Sigma^{0.5}_{s} \Sigma_{t}
         \Sigma^{0.5}_{s})^{0.5} \Sigma^{-0.5}_{s} + P/G^{*} I

        (19) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{+0.5}_{t} (\Sigma^{0.5}_{t} \Sigma_{s}
         \Sigma^{0.5}_{t})^{0.5} \Sigma^{-0.5}_{t} \Sigma^{\dagger}_{s} + P/G^{*} I

         \Sigma_w = \Sigma^{0.5}_{t} (I - \Sigma^{0.5}_{t} T^{*} \Sigma^{\dagger}_{s} T^{*}
          \Sigma^{0.5}_{t}) \Sigma^{0.5}_{t},  T^{*} = T_{t \longrightarrow s} (17)

    [1] D. Freirich, T. Michaeli and R. Meir.
    `A Theory of the Distortion-Perception Tradeoff in Wasserstein Space <https://proceedings.neurips.cc/paper/2021/
    hash/d77e68596c15c53c2a33ad143739902d-Abstract.html>`_
    """

    INTERNAL_STATES = [
        '_running_sum_source',
        '_running_sum_target',
        '_running_sum_cov_source',
        '_running_sum_cov_target',
        '_n_obs_source',
        '_n_obs_target',
    ]

    OPERATORS = [
        '_transport_operator',
        '_cov_stochastic_noise'
    ]

    def __init__(self,
                 dim: int,
                 diag: bool,
                 pg_star: float = 0.,
                 stochastic: bool = False,
                 persistent: bool = False,
                 make_pd: bool = False,
                 verbose: bool = False
                 ):
        r"""
        :param dim: The dimensionality of the distributions to transport
        :param diag: If ``True`` will suppose the samples come from isotropic distributions (with diagonal covariance)
        :param pg_star: Perception-distortion ratio. can be seen as temperature.
        :param stochastic: If ``True`` return (T_{s -> t}, \Sigma_w) of (19) else return (T_{s -> t}, `None`) (17)
        :param persistent: whether the buffers are part of this module's :attr:`state_dict`.
        :param make_pd: Add the minimum eigenvalue in order to make matrices pd, if needed
        """
        super().__init__()
        self.dim = dim
        self.diag = diag
        self.pg_star = pg_star
        self.stochastic = stochastic
        self.make_pd = make_pd
        self.verbose = verbose
        self._reduce = partial(sync_ddp_if_available, reduce_op="sum")

        vec_init = torch.zeros(self.dim, dtype=torch.double)
        mat_init = vec_init if diag else torch.zeros(self.dim, self.dim, dtype=torch.double)

        # internal states
        self.register_buffer("_running_sum_source", vec_init.clone(), persistent=persistent)
        self.register_buffer("_running_sum_target", vec_init.clone(), persistent=persistent)
        self.register_buffer("_running_sum_cov_source", mat_init.clone(), persistent=persistent)
        self.register_buffer("_running_sum_cov_target", mat_init.clone(), persistent=persistent)
        self.register_buffer("_n_obs_source", torch.zeros([], dtype=torch.long), persistent=persistent)
        self.register_buffer("_n_obs_target", torch.zeros([], dtype=torch.long), persistent=persistent)

        # operators
        self.register_buffer("_mean_source", vec_init.clone(), persistent=persistent)
        self.register_buffer("_mean_target", vec_init.clone(), persistent=persistent)
        self.register_buffer("_cov_source", mat_init.clone(), persistent=persistent)
        self.register_buffer("_cov_target", mat_init.clone(), persistent=persistent)
        self.register_buffer("_transport_operator", mat_init.clone(), persistent=persistent)
        self.register_buffer("_cov_stochastic_noise", mat_init.clone(), persistent=persistent)

    def update(self, source_samples: Optional[Tensor] = None, target_samples: Optional[Tensor] = None) -> None:
        # noinspection Duplicates
        if source_samples is not None:
            flattened = source_samples.flatten(1).double().to(self._running_sum_source.device)
            if flattened.size(1) != self.dim:
                ValueError(f"`source_samples` flattened is expected to have dimensionality equaled to {self.dim}")
            self._n_obs_source += flattened.size(0)
            self._running_sum_source += flattened.sum(0)
            self._running_sum_cov_source += (flattened ** 2).sum(0) if self.diag else flattened.T.mm(flattened)

        # noinspection Duplicates
        if target_samples is not None:
            flattened = target_samples.flatten(1).double().to(self._running_sum_target.device)
            if flattened.size(1) != self.dim:
                ValueError(f"`target_samples` flattened is expected to have dimensionality equaled to {self.dim}")
            self._n_obs_target += flattened.size(0)
            self._running_sum_target += flattened.sum(0)
            self._running_sum_cov_target += (flattened ** 2).sum(0) if self.diag else flattened.T.mm(flattened)

    def compute(self) -> Tensor:
        # reduce states across GPUs
        for buffer in self.INTERNAL_STATES: setattr(self, buffer, self._reduce(getattr(self, buffer)))

        self._mean_source, self._cov_source = mean_cov(  # noqa
            self._running_sum_source, self._running_sum_cov_source, self._n_obs_source, diag=self.diag
        )
        self._mean_target, self._cov_target = mean_cov(  # noqa
            self._running_sum_target, self._running_sum_cov_target, self._n_obs_target, diag=self.diag
        )

        w2 = w2_gaussian(
            self._mean_source,
            self._mean_target,
            torch.diag_embed(self._cov_source) if self.diag else self._cov_source,
            torch.diag_embed(self._cov_target) if self.diag else self._cov_target,
            make_pd=self.make_pd,
            verbose=self.verbose
        )

        self._transport_operator, self._cov_stochastic_noise = compute_transport_operators(  # noqa
            self._cov_source,
            self._cov_target,
            stochastic=self.stochastic,
            diag=self.diag,
            pg_star=self.pg_star,
            make_pd=self.make_pd,
            verbose=self.verbose
        )
        return w2

    def transport(self, inputs: Tensor) -> Tensor:
        flattened = inputs.flatten(1).double().to(self._mean_source.device)
        if flattened.size(1) != self.dim:
            ValueError(f"`inputs` flattened is expected to have dimensionality equaled to {self.dim}")

        transported = apply_transport(
            flattened,
            self._mean_source.type_as(flattened),
            self._mean_target.type_as(flattened),
            self._transport_operator.type_as(flattened),
            self._cov_stochastic_noise.type_as(flattened),
            self.diag
        )

        return transported.to(dtype=inputs.dtype).unflatten(1, inputs.shape[1:])

    def sample(self, shape, dtype, device, from_dist='source') -> Tensor:
        if np.prod(shape[1:]) != self.dim:
            ValueError(f"`shape` is expected to have dimensionality equaled to {self.dim}")

        mean, cov = getattr(self, f'_mean_{from_dist}'), getattr(self, f'_cov_{from_dist}')
        if self.make_pd:
            cov = torch.clamp(cov, min=STABILITY_CONST) if self.diag else make_psd(cov, strict=True)
        dist = Normal(mean, scale=cov ** 0.5) if self.diag else MultivariateNormal(mean, covariance_matrix=cov)
        samples = dist.sample((shape[0],)).unflatten(1, shape[1:]).to(dtype=dtype, device=device)
        return samples

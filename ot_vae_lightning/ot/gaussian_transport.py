"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of Wasserstein 2 Optimal transport module

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import numpy as np
from typing import Optional
import torch
from torch import Tensor
from torchmetrics import Metric
from torch.distributions import MultivariateNormal, Normal

from ot_vae_lightning.ot.w2_utils import w2_gaussian, compute_transport_operators, apply_transport
from ot_vae_lightning.ot.matrix_utils import make_psd, STABILITY_CONST

__all__ = ['GaussianTransport']


class GaussianTransport(Metric):
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

    We inherit from torchmetrics.Metric class to automatically handle distributed synchronization.
    """
    full_state_update = False

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
        super().__init__(dist_sync_on_step=False)
        self.dim = dim
        self.diag = diag
        self.pg_star = pg_star
        self.stochastic = stochastic
        self.make_pd = make_pd
        self.verbose = verbose

        vec_init = torch.zeros(self.dim, dtype=torch.double)
        mat_init = vec_init if diag else torch.zeros(self.dim, self.dim, dtype=torch.double)
        self.add_state("mean_source", vec_init.clone(), dist_reduce_fx='sum', persistent=persistent)
        self.add_state("mean_target", vec_init.clone(), dist_reduce_fx='sum', persistent=persistent)
        self.add_state("cov_source", mat_init.clone(), dist_reduce_fx='sum', persistent=persistent)
        self.add_state("cov_target", mat_init.clone(), dist_reduce_fx='sum', persistent=persistent)
        self.add_state("n_obs_source", torch.zeros([], dtype=torch.long), dist_reduce_fx='sum', persistent=persistent)
        self.add_state("n_obs_target", torch.zeros([], dtype=torch.long), dist_reduce_fx='sum', persistent=persistent)
        self.add_state("transport_operator", mat_init.clone(), persistent=persistent)
        if self.stochastic: self.add_state("cov_stochastic_noise", mat_init.clone(), persistent=persistent)
        else: self.cov_stochastic_noise = None

    def update(self, source_samples: Optional[Tensor] = None, target_samples: Optional[Tensor] = None) -> None:
        # noinspection Duplicates
        if source_samples is not None:
            flattened = source_samples.flatten(1).double()
            if flattened.size(1) != self.dim:
                ValueError(f"`source_samples` flattened is expected to have dimensionality equaled to {self.dim}")
            self.n_obs_source += flattened.size(0)
            self.mean_source += flattened.sum(0)
            self.cov_source += (flattened ** 2).sum(0) if self.diag else flattened.T.mm(flattened)

        # noinspection Duplicates
        if target_samples is not None:
            flattened = target_samples.flatten(1).double()
            if flattened.size(1) != self.dim:
                ValueError(f"`target_samples` flattened is expected to have dimensionality equaled to {self.dim}")
            self.n_obs_target += flattened.size(0)
            self.mean_target += flattened.sum(0)
            self.cov_target += (flattened ** 2).sum(0) if self.diag else flattened.T.mm(flattened)

    def _compute_helper(self) -> Tensor:
        self.mean_source /= self.n_obs_source
        self.mean_target /= self.n_obs_target
        self.cov_source /= self.n_obs_source
        self.cov_target /= self.n_obs_target

        if self.diag:
            self.cov_source -= self.mean_source ** 2
            self.cov_target -= self.mean_target ** 2
        else:
            self.cov_source -= self.mean_source.unsqueeze(-1) @ self.mean_source.unsqueeze(0)
            self.cov_target -= self.mean_target.unsqueeze(-1) @ self.mean_target.unsqueeze(0)

        w2 = w2_gaussian(
            self.mean_source,
            self.mean_target,
            torch.diag_embed(self.cov_source) if self.diag else self.cov_source,
            torch.diag_embed(self.cov_target) if self.diag else self.cov_target,
            make_pd=self.make_pd,
            verbose=self.verbose
        )

        self.transport_operator, self.cov_stochastic_noise = compute_transport_operators(
            self.cov_source,
            self.cov_target,
            stochastic=self.stochastic,
            diag=self.diag,
            pg_star=self.pg_star,
            make_pd=self.make_pd,
            verbose=self.verbose
        )
        return w2

    def compute(self) -> Tensor:
        return self._compute_helper()

    def transport(self, inputs: Tensor) -> Tensor:
        flattened = inputs.flatten(1).double()
        if flattened.size(1) != self.dim:
            ValueError(f"`inputs` flattened is expected to have dimensionality equaled to {self.dim}")

        transported = apply_transport(
            flattened,
            self.mean_source.type_as(flattened),
            self.mean_target.type_as(flattened),
            self.transport_operator.type_as(flattened),
            self.cov_stochastic_noise.type_as(flattened) if self.cov_stochastic_noise is not None else None,
            self.diag
        )

        return transported.to(dtype=inputs.dtype).unflatten(1, inputs.shape[1:])

    def sample(self, shape, dtype, device, from_dist='source') -> Tensor:
        if np.prod(shape[1:]) != self.dim:
            ValueError(f"`shape` is expected to have dimensionality equaled to {self.dim}")

        mean, cov = getattr(self, f'mean_{from_dist}'), getattr(self, f'cov_{from_dist}')
        if self.make_pd:
            cov = torch.clamp(cov, min=STABILITY_CONST) if self.diag else make_psd(cov, strict=True)
        dist = Normal(mean, scale=cov ** 0.5) if self.diag else MultivariateNormal(mean, covariance_matrix=cov)
        samples = dist.sample((shape[0],)).unflatten(1, shape[1:]).to(dtype=dtype, device=device)
        return samples

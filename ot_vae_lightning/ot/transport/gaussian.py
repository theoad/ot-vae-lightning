"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of Wasserstein 2 optimal gaussian transport

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Union, Literal

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal
from torch.types import _size, _dtype

from ot_vae_lightning.ot.transport.base import TransportOperator
from ot_vae_lightning.ot.w2_utils import w2_gaussian, compute_transport_operators, apply_transport
from ot_vae_lightning.ot.gaussian_model import GaussianModel

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

    def __init__(self,
                 dim: int,
                 batch_shape: _size,
                 dtype: _dtype = torch.double,
                 diag: bool = False,
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
        super().__init__(dim, batch_shape, dtype)
        self.dim = dim
        self.diag = diag
        self.pg_star = pg_star
        self.stochastic = stochastic
        self.make_pd = make_pd
        self.verbose = verbose
        self.gaussian_source = GaussianModel(batch_shape, dim, dtype, diag, persistent)
        self.gaussian_target = GaussianModel(batch_shape, dim, dtype, diag, persistent)

        self.transport_operator = None
        self.cov_stochastic_noise = None

    def reset(self) -> None:
        self.gaussian_source.reset()
        self.gaussian_target.reset()
        self.transport_operator = None
        self.cov_stochastic_noise = None

    @property
    def source_distribution(self) -> Union[Normal, MultivariateNormal]:
        return self.gaussian_source.distribution

    @property
    def target_distribution(self) -> Union[Normal, MultivariateNormal]:
        return self.gaussian_target.distribution

    def _update(self, samples: Tensor, dist: Literal['source', 'target']):
        getattr(self, f'gaussian_{dist}')(samples)

    def compute(self) -> Tensor:
        # reduce states across GPUs under the hood
        self.gaussian_source.fit()
        self.gaussian_target.fit()

        w2 = w2_gaussian(
            self.gaussian_source.mean,
            self.gaussian_target.mean,
            torch.diag_embed(self.gaussian_source.cov) if self.diag else self.gaussian_source.cov,
            torch.diag_embed(self.gaussian_target.cov) if self.diag else self.gaussian_target.cov,
            make_pd=self.make_pd,
            verbose=self.verbose
        )

        self.transport_operator, self.cov_stochastic_noise = compute_transport_operators(
            self.gaussian_source.cov,
            self.gaussian_target.cov,
            stochastic=self.stochastic,
            diag=self.diag,
            pg_star=self.pg_star,
            make_pd=self.make_pd,
            verbose=self.verbose
        )
        return w2

    def transport(self, inputs: Tensor) -> Tensor:
        if inputs.size(-1) != self.dim:
            raise ValueError("`inputs` dimensionality must match the model dimensionality")
        if inputs.shape[:-2] != self.batch_shape and inputs.shape[:-1] != self.batch_shape:
            raise ValueError("`inputs` leading dims must match the model batch_shape with optional trailing batch dimensions")
        is_batched = inputs.dim() == len(self.batch_shape) + 2

        transported = apply_transport(
            inputs,
            self.gaussian_source.mean.unsqueeze(-2) if is_batched else self.gaussian_source.mean,
            self.gaussian_target.mean.unsqueeze(-2) if is_batched else self.gaussian_target.mean,
            self.transport_operator.unsqueeze(-3 + bool(self.diag)) if is_batched else self.transport_operator,
            self.cov_stochastic_noise.unsqueeze(-3 + bool(self.diag)) if is_batched else self.cov_stochastic_noise,
            self.diag
        )

        return transported.to(dtype=inputs.dtype)

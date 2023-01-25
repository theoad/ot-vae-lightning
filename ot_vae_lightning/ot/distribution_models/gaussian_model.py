"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a Multivariate Gaussian Model (MVG)

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Union, Optional, Tuple

import torch
from torch import Tensor
import torch.distributions as D
import torch.nn as nn
import torch.nn.utils.parametrize as P

from ot_vae_lightning.ot.distribution_models.base import DistributionModel
from ot_vae_lightning.ot.w2_utils import W2Mixin
from ot_vae_lightning.ot.matrix_utils import eye_like, make_psd

__all__ = ['GaussianModel']


class GaussianModel(DistributionModel, W2Mixin):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a Multivariate Gaussian Model (MVG)

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad

    """
    Distribution = Union[D.Independent, D.MultivariateNormal]

    def __init__(
            self,
            *size: int,
            w2_cfg={},
            **kwargs,
    ):
        DistributionModel.__init__(self, *size, **kwargs)
        W2Mixin.__init__(self, **w2_cfg)
        self.batch_dim = -2
        self.register_buffer("cov_init", torch.ones_like(self.vec_init) if self.diag else eye_like(self.mat_init))
        self.mean = nn.Parameter(self.vec_init.clone(), requires_grad=self.update_with_autograd)
        self.cov = nn.Parameter(self.cov_init.clone(), requires_grad=self.update_with_autograd)

        if self.update_with_autograd:
            # Cholesky decomposition of the covariance matrices with Exponent+tril reparametrization
            # This ensures covariance validity while optimizing
            P.register_parametrization(self, "cov", ExpScaleTril(diag=self.diag))
        else:
            self.register_buffer("_running_sum", torch.zeros_like(self.mean.data))
            self.register_buffer("_running_sum_cov", torch.zeros_like(self.cov.data))
            self.register_buffer("_n_obs", torch.zeros(self.vec_shape[:-1], dtype=self.dtype))
            # This parametrization ensures the covariance is always symmetric and strictly positive definite
            P.register_parametrization(self, "cov", Symmetric(diag=self.diag))
            P.register_parametrization(self, "cov", MakePositiveDefinite(diag=self.diag, strict=True))

    @torch.no_grad()
    def reset(self) -> None:
        self._update_mean(self.vec_init)
        self._update_cov(self.cov_init)

        if not self.update_with_autograd:
            self._running_sum.zero_()
            self._running_sum_cov.zero_()
            self._n_obs.zero_()

    @property
    def distribution(self) -> Distribution:
        return self.instantiate_normal(
            self.mean,
            scale=self.cov ** 0.5,
            scale_tril=self.cov if self.update_with_autograd else None,
            covariance_matrix=self.cov if not self.update_with_autograd else None
        )

    @property
    def batched_distribution(self) -> Distribution:
        return self.instantiate_normal(
            self.mean.unsqueeze(self.batch_dim),
            scale=self.cov.unsqueeze(self.batch_dim) ** 0.5,
            scale_tril=self.cov.unsqueeze(self.batch_dim-1) if self.update_with_autograd and not self.diag else None,
            covariance_matrix=self.cov.unsqueeze(self.batch_dim-1) if not self.update_with_autograd and not self.diag else None,
        )

    @property
    def variances(self) -> Tensor:
        return self.get_var_normal(self.distribution)

    @torch.no_grad()
    def update(self, samples: Tensor) -> None:
        self._update_warn()
        self._validate_samples(samples)
        samples = samples.detach().requires_grad_(False).type_as(self._running_sum)
        n_obs, sum, sum_cov = self._stats(samples, reduce=self.reduce_on_update)

        self._n_obs = self.ema_update(self._n_obs, n_obs)  # noqa
        self._running_sum = self.ema_update(self._running_sum, sum)  # noqa
        self._running_sum_cov = self.ema_update(self._running_sum_cov, sum_cov)  # noqa

    @torch.no_grad()  # must be with no_grad because we assign parameters value by hand here
    def fit(self, samples: Optional[Tensor] = None) -> None:
        self._fit_warn()

        if self.update_with_autograd:
            if samples is None: return
            mean, cov, seen = self._compute_mean_cov(*self._stats(samples, reduce=True))
            self._update_mean(mean, seen)
            self._update_cov(cov, seen)

        if samples is not None:
            self.update(samples)

        self._n_obs, self._running_sum, self._running_sum_cov = self._stats(None, reduce=True)  # noqa
        mean, cov, seen = self._compute_mean_cov(self._n_obs, self._running_sum, self._running_sum_cov)
        self._update_mean(mean, seen)
        self._update_cov(cov, seen)

    def predict(self, samples: Tensor) -> Tensor:
        self._validate_samples(samples)
        samples = samples.type_as(self.mean)
        return self.batched_distribution.log_prob(samples)

    def w2(self, other: Distribution) -> Tensor:
        return self.w2_gaussian(
            self.mean,
            other.mean,
            self.variances,
            self.get_var_normal(other)
        )

    def extra_repr(self) -> str:
        return super().extra_repr() + W2Mixin.__repr__(self)

    def _stats(self, samples: Optional[Tensor], reduce=True):
        if samples is not None:
            n_obs = torch.as_tensor(samples.size(-2), dtype=samples.dtype, device=samples.device)
            samples_sum = samples.sum(-2)
            samples_cov_sum = (samples ** 2).sum(-2) if self.diag else torch.einsum("...bi,...bj->...ij", samples, samples)
        else:
            n_obs = self._n_obs
            samples_sum = self._running_sum
            samples_cov_sum = self._running_sum_cov
        if reduce:
            n_obs = self.reduce(n_obs)
            samples_sum = self.reduce(samples_sum)
            samples_cov_sum = self.reduce(samples_cov_sum)
        return n_obs, samples_sum, samples_cov_sum

    def _compute_mean_cov(
            self, n_obs: Tensor, sum: Tensor, sum_cov: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        if (n_obs.dim() == 0 and n_obs == 0) or (n_obs == 0).all(): return None, None, None
        seen = n_obs > 1e-8
        mean, cov = self.mean_cov(sum[seen], sum_cov[seen], n_obs[seen])
        return mean, cov, seen

    def _update_mean(self, val: Optional[Tensor], seen: Optional[Tensor] = None):
        if val is None: return
        if seen is None:
            self.mean.copy_(val.type_as(self.mean))
        else:
            # TODO: This may lead to silent bug (autograd doesn't track accesses to .data)
            self.mean.data[seen] = val.type_as(self.mean)

    def _update_cov(self, val: Optional[Tensor], seen: Optional[Tensor] = None):
        if val is None: return
        if seen is None:
            # uses parametrization right_inverse under the hood
            self.cov = val.type_as(self.cov)
        else:
            tmp = self.cov
            tmp[seen] = val
            self.cov = tmp.type_as(tmp)


class ExpScaleTril(nn.Module):
    """
    Parametrization to ensure scale parameters satisfy variance / covariance constrains
    It maps the reparametrized matrix to be a lower triangular matrix with positive diagonal
    (e.g. to represent the cholesky decomposition of a SPD matrix).
    """
    def __init__(self, diag):
        super().__init__()
        self.diag = diag

    def forward(self, x: Tensor) -> Tensor:
        if self.diag: return x.exp()
        return x.tril(-1) + torch.diag_embed(x.diagonal(dim1=-1, dim2=-2).exp())

    def right_inverse(self, x: Tensor) -> Tensor:
        return x if self.diag else x.tril()


class MakePositiveDefinite(nn.Module):
    """
    Parametrization to add a small value on the diagonal of the input tensor to make sure it is PD
    """
    def __init__(self, diag, strict):
        super().__init__()
        self.diag = diag
        self.strict = strict

    def forward(self, x):
        return make_psd(x, strict=self.strict, return_correction=False, diag=self.diag)

    def right_inverse(self, x):
        return x


class Symmetric(nn.Module):
    def __init__(self, diag):
        super().__init__()
        self.diag = diag

    def forward(self, X):
        return X if self.diag else X.triu() + X.triu(1).transpose(-1, -2)

    def right_inverse(self, X):
        return X

"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of Gaussian Mixture Model (GMM)

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D
import torch.nn.utils.parametrize as P

from ot_vae_lightning.ot.distribution_models.base import MixtureMixin
from ot_vae_lightning.ot.distribution_models.gaussian_model import GaussianModel
from ot_vae_lightning.ot.distribution_models.codebook_model import CodebookModel
from ot_vae_lightning.ot.w2_utils import W2Mixin


class GaussianMixtureModel(GaussianModel, CodebookModel):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of Gaussian Mixture Model (GMM)
    
    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved
    
    .. warning:: Work in progress. This implementation is still being verified.
    
    .. _TheoA: https://github.com/theoad
    """
    Distribution = D.MixtureSameFamily

    def __init__(
            self,
            *size: int,
            mixture_cfg = {},
            **kwargs,
    ):
        # we init grand-father and not CodebookModel on purpose
        # we don't actually need `self.codebook` variable as well as _running_sum and _n_obs buffers
        # which already exist in GaussianModel
        MixtureMixin.__init__(self, *size[:-1], **mixture_cfg)
        GaussianModel.__init__(self, *size, **kwargs)
        self.batch_dim = -3
        self.register_buffer("weight_init", self._weight_init.type_as(self.vec_init))
        self._weights = nn.Parameter(self.weight_init.clone(), requires_grad=self.update_with_autograd)

        if self.update_with_autograd:
            P.register_parametrization(self, "_weights", nn.Softmax(-1))
        else:
            P.register_parametrization(self, "_weights", NormSum(1.))

    @property
    def weights(self):
        return self._weights

    @property
    def vec_shape(self):
        return *self.leading_shape, self.n_components, self.dim

    def reset(self) -> None:
        GaussianModel.reset(self)
        self._weights = self.weight_init

    @property
    def distribution(self) -> Distribution:
        return D.MixtureSameFamily(D.Categorical(self.weights), super().distribution)

    @property
    def batched_distribution(self) -> Distribution:
        return D.MixtureSameFamily(D.Categorical(self.weights.unsqueeze(-2)), super().batched_distribution)

    @property
    def variances(self) -> Tensor:
        return self.get_var_normal(self.distribution.component_distribution)

    @property
    def batched_variances(self) -> Tensor:
        return self.get_var_normal(self.batched_distribution.component_distribution)

    def update(self, samples: Tensor) -> None:
        return CodebookModel.update(self, samples)

    def fit(self, samples: Optional[Tensor] = None) -> None:
        return CodebookModel.fit(self, samples)

    def energy(self, samples: Tensor) -> Tensor:
        self._validate_samples(samples)
        dist = self.batched_distribution
        samples = samples.type_as(dist.mean)
        if self.batched_distribution._validate_args: dist._validate_sample(samples)  # noqa
        samples = dist._pad(samples)  # noqa [*leading_shape, B, 1, dim]
        log_prob_x = dist.component_distribution.log_prob(samples)  # [*leading_shape, B, comp]
        log_mix_prob = torch.log_softmax(dist.mixture_distribution.logits, dim=-1)  # [*leading_shape, B, comp]
        return (log_prob_x + log_mix_prob).type_as(samples)  # [*leading_shape, B, comp]

    def predict_mean_var(self, assignments: Tensor) -> Tuple[Tensor, Tensor]:
        # [*batch_dim, B, comp] x [*batch_dim, comp, dim] --> [*batch_dim, B, dim]
        mean = assignments.type_as(self.mean) @ self.mean
        var = assignments.type_as(self.variances) @ (self.variances if self.diag else self.variances.flatten(-2))
        if not self.diag:
            var = var.unflatten(-1, (self.dim, self.dim))  # [*batch_dim, B, dim * dim] --> [*batch_dim, B, dim, dim]
        return mean.type_as(assignments), var.type_as(assignments)

    def kmean_iteration(self, samples: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        if samples is None:
            return self._n_obs, self._running_sum, self._running_sum_cov

        samples_cov = (samples ** 2) if self.diag else (samples.unsqueeze(-1) @ samples.unsqueeze(-2)).flatten(-2)

        weights, _, _ = self.assign(samples)  # [*, b, n_comp]
        weights_sum = weights.sum(-2)  # [*, n_comp]
        samples_weighted_sum = weights.transpose(-1, -2) @ samples  # [*, n_comp, b] x [*, b, d] --> [*, n_comp, d]
        samples_weighted_cov_sum = (weights.transpose(-1, -2) @ samples_cov)
        if not self.diag: samples_weighted_cov_sum = samples_weighted_cov_sum.unflatten(-1, (self.dim, self.dim))

        return weights_sum, samples_weighted_sum, samples_weighted_cov_sum

    def w2(self, other: Distribution) -> Tensor:
        total_cost, _ = self.batch_ot_gmm(
            self.mean,
            other.component_distribution.mean,
            self.variances,
            self.get_var_normal(other.component_distribution),
            weight_source=self.weights,
            weight_target=other.mixture_distribution.probs,
            max_iter=100
        )
        return total_cost

    def extra_repr(self) -> str:
        return super().extra_repr() + W2Mixin.__repr__(self) + MixtureMixin.__repr__(self)

    def _update_weights(self, val: Optional[Tensor], seen: Optional[Tensor] = None):
        if val is None: return
        if seen is None:
            # uses parametrization right_inverse under the hood
            self._weights = val.type_as(self._weights)
        else:
            tmp = self._weights
            tmp[seen] = val
            self._weights = tmp.type_as(tmp)

    def _update_parameters(self, *kmeans_iter_res):
        weights_sum, samples_sum, samples_cov_sum = kmeans_iter_res
        mean, cov, seen = self._compute_mean_cov(self.laplace_smoothing(weights_sum), samples_sum, samples_cov_sum)
        self._update_mean(mean, seen)
        self._update_cov(cov, seen)
        self._update_weights(weights_sum[seen], seen)

    def _update_buffers(self, *kmeans_iter_res, decay=False):
        weights_sum, samples_sum, samples_cov_sum = kmeans_iter_res
        hit = weights_sum > 1e-8

        if decay:
            self._n_obs[hit] = self.ema_update(self._n_obs[hit], weights_sum[hit])
            self._running_sum[hit] = self.ema_update(self._running_sum[hit], samples_sum[hit])
            self._running_sum_cov[hit] = self.ema_update(self._running_sum_cov[hit], samples_cov_sum[hit])
        else:
            self._n_obs[hit] = weights_sum[hit]
            self._running_sum[hit] = samples_sum[hit]
            self._running_sum_cov[hit] = samples_cov_sum[hit]

        return self._n_obs, self._running_sum, self._running_sum_cov

    def _init_parameters(self, samples: Tensor) -> None:
        if torch.allclose(self.mean, self.vec_init):
            rand_indices = torch.randperm(samples.size(-2))[:self.n_components]
            self._update_mean(samples[..., rand_indices, :])
            self._n_obs += 1


class NormSum(nn.Module):
    def __init__(self, val=1.):
        super().__init__()
        self.val = val

    def forward(self, X):
        return self.val * X / X.sum(-1, keepdim=True)

    def right_inverse(self, X):
        return X

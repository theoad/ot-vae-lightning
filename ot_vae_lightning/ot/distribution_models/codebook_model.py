"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a Discrete Codebook Model

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.distributions as D
from torch import Tensor
import torch.nn.functional as F
from ot_vae_lightning.ot.distribution_models.base import DistributionModel, MixtureMixin
from ot_vae_lightning.ot.w2_utils import sinkhorn_log

__all__ = ['CodebookModel', 'CategoricalEmbeddings']


class CategoricalEmbeddings(D.Categorical):
    def __init__(
            self,
            embeddings: Tensor,
            probs: Optional[Tensor] = None,
            logits: Optional[Tensor] = None,
    ) -> None:
        super().__init__(probs, logits)
        self.embeddings = embeddings
        if self.probs.shape != self.embeddings.shape[:-1]:
            raise ValueError("`probs` and `embeddings` should have the same leading dimensions")

    def _one_hot(self, indices: Tensor) -> Tensor:
        return F.one_hot(indices, self._num_events).type_as(indices)

    def _select(self, weights: Tensor) -> Tensor:
        return (weights.unsqueeze(-2).type_as(self.embeddings) @ self.embeddings).squeeze(-2)

    def expand(self, batch_shape, _instance=None):
        new = super().expand(batch_shape, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        new.embeddings = new.embeddings.expand(param_shape)
        return new

    @property
    def mean(self):
        return self._select(self.probs)

    @property
    def mode(self):
        return self._select_one_hot(self.probs.argmax(-1))

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self._select_one_hot(super().sample(sample_shape))

    def _select_one_hot(self, index_list: Tensor) -> Tensor:
        return self._select(self._one_hot(index_list))


class CodebookModel(DistributionModel, MixtureMixin):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a Discrete Codebook Model

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """

    Distribution = CategoricalEmbeddings

    def __init__(
            self,
            *size: int,
            mixture_cfg = {},
            **kwargs
    ) -> None:
        MixtureMixin.__init__(self, *size[:-1], **mixture_cfg)
        DistributionModel.__init__(self, *size, **kwargs)
        self.register_buffer("weight_init", self._weight_init.type_as(self.vec_init))
        self.codebook = nn.Parameter(self.vec_init.clone(), requires_grad=self.update_with_autograd)

        if not self.update_with_autograd:
            self.register_buffer("_running_sum", torch.zeros_like(self.vec_init.clone()))
            self.register_buffer("_n_obs", torch.zeros(*self.leading_shape, self.n_components).type_as(self.vec_init))

    @property
    def weights(self):
        if not hasattr(self, '_n_obs') or torch.allclose(self._n_obs, torch.zeros_like(self._n_obs)):
            return self.weight_init.type_as(self.codebook)
        else:
            return self._n_obs.type_as(self.codebook) / self._n_obs.sum(-1, keepdim=True)

    @property
    def vec_shape(self):
        return *self.leading_shape, self.n_components, self.dim

    @torch.no_grad()
    def reset(self) -> None:
        self.codebook.copy_(self.vec_init)

        if not self.update_with_autograd:
            self._running_sum.zero_()
            self._n_obs.zero_()

    @property
    def distribution(self) -> Distribution:
        return CategoricalEmbeddings(self.codebook, probs=self.weights)

    @property
    def batched_distribution(self) -> Distribution:
        return CategoricalEmbeddings(self.codebook.unsqueeze(-3), probs=self.weights.unsqueeze(-2))

    @torch.no_grad()
    def update(self, samples: Tensor) -> None:
        self._update_warn()
        self._validate_samples(samples)
        samples = samples.detach().requires_grad_(False).type_as(self._running_sum)

        self._init_parameters(samples)
        kmean_res = self.kmean_iteration(samples)
        if self.reduce_on_update: kmean_res = [self.reduce(res) for res in kmean_res]
        buffers = self._update_buffers(*kmean_res, decay=True)
        self._update_parameters(*buffers)

    @torch.no_grad()
    def fit(self, samples: Optional[Tensor] = None) -> None:
        self._fit_warn()

        if samples is not None:
            self._validate_samples(samples)
            samples = samples.detach().requires_grad_(False).type_as(self._running_sum)
            self._init_parameters(samples)

        for _ in range(self.kmeans_iter):
            kmean_res = self.kmean_iteration(samples)
            self._update_parameters(*[self.reduce(res) for res in kmean_res])

        if self.kmeans_iter > 0:
            self._update_buffers(*kmean_res, decay=False)  # noqa

    def predict(self, features: Tensor) -> Tuple[Tensor, Tensor, D.Categorical]:
        weights, indices, distribution = self.assign(features)  # [*, b, n_comp]
        preds: Tensor = weights @ self.codebook  # [*, b, n_comp] x [*, n_comp, d] --> [*, b, d]
        return preds, indices, distribution

    def energy(self, samples: Tensor) -> Tensor:
        self._validate_samples(samples)
        samples = samples.type_as(self.codebook)  # [*, b, d]
        if self.metric == 'euclidean':
            cdist = torch.cdist(samples, self.codebook, self.p)       # [*, b, n_comp]
            return 1 / (cdist + 1e-8)
        elif self.metric == 'cosine':
            norm_x = samples.abs().pow(self.p).sum(-1, keepdim=True)        # [*, b, 1]
            norm_c = self.codebook.abs().pow(self.p).sum(-1).unsqueeze(-2)  # [*, 1, n_comp]
            dot_product = (samples @ self.codebook.transpose(-2, -1)).abs()   # [*, b, n_comp, d] x [*, b, d, n_comp]
            cosine_similarity = dot_product / (norm_x * norm_c + 1e-8) ** (1 / self.p)
            return cosine_similarity
        else:
            raise NotImplementedError(f"Supported `metric`: 'cosine', 'euclidean'. Got `metric`={self.metric}")

    def kmean_iteration(self, samples: Optional[Tensor]) -> Tuple[Tensor, ...]:
        if samples is None:
            return self._n_obs, self._running_sum
        else:
            weights_sum, samples_weighted_sum = super().kmean_iteration(samples)
            return weights_sum, samples_weighted_sum

    def w2(self, other: Distribution) -> Tensor:
        cost = 1 / (self.energy(other.embeddings) + 1e-8)
        transport_matrix = sinkhorn_log(
            self.distribution.probs, other.probs, cost,
            reg=1e-5, max_iter=100, threshold=1e-3
        )
        total_cost = torch.sum(cost * transport_matrix, dim=(-2, -1))
        return total_cost

    def extra_repr(self) -> str:
        return DistributionModel.extra_repr(self) + ", " + MixtureMixin.extra_repr(self)

    def _update_parameters(self, *kmeans_iter_res: Tensor) -> None:
        weights_sum, samples_sum = kmeans_iter_res

        # we only update codebook components which were observed (those for whom the weights_sum > 0)
        hit = weights_sum > 1e-8
        self.codebook.data[hit] = samples_sum[hit] / self.laplace_smoothing(weights_sum[hit]).unsqueeze(-1)

    def _update_buffers(self, *kmeans_iter_res: Tensor, decay: bool = False):
        weights_sum, samples_sum = kmeans_iter_res

        # we only update codebook components which were observed (those for whom the weights_sum > 0)
        hit = weights_sum > 1e-8
        if decay:
            self._n_obs[hit] = self.ema_update(self._n_obs[hit], weights_sum[hit])
            self._running_sum[hit] = self.ema_update(self._running_sum[hit], samples_sum[hit])
        else:
            self._n_obs[hit] = weights_sum[hit]
            self._running_sum[hit] = samples_sum[hit]

        return self._n_obs, self._running_sum

    def _init_parameters(self, samples: Tensor) -> None:
        if torch.allclose(self.codebook, self.vec_init):
            rand_indices = torch.randperm(samples.size(-2))[:self.n_components]
            self.codebook.copy_(samples[..., rand_indices, :])
            self._n_obs += 1

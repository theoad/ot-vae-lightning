"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of Gaussian Mixture Model (GMM)

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Tuple
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from ot_vae_lightning.utils.elr import EqualizedLR
from accelerate import Accelerator
from torch.types import _size, _dtype


class ExpScaleTril(nn.Module):
    def __init__(self, diag):
        super().__init__()
        self.diag = diag

    def forward(self, x: Tensor) -> Tensor:
        if self.diag: return x.exp()
        return x.tril(-1) + torch.diag_embed(x.diagonal(dim1=-1, dim2=-2).exp())


class TensorDatasetDim(TensorDataset):
    def __init__(self, tensors: Tensor, dim=0):
        super().__init__(tensors)
        self.dim = dim

    def __getitem__(self, index):
        return self.tensors[0].select(dim=self.dim, index=index).unsqueeze(self.dim)


class GaussianMixtureModel(nn.Module):
    def __init__(
            self,
            batch_shape: _size,
            num_components: int,
            dim: int,
            dtype: _dtype,
            diag: bool,
            gain=1.):
        super().__init__()
        self.batch_shape = batch_shape
        self.num_components = num_components
        self.dim = dim
        self.diag = diag
        self.dtype = dtype

        self.means = nn.Parameter(torch.randn(*batch_shape, num_components, dim, dtype=dtype))
        parametrize.register_parametrization(self, 'means', EqualizedLR(fan_in_dims=[-1,-2], gain=gain))

        # Cholesky decomposition of the covariance matrices with Exp+tril reparametrization to ensure validity
        init_log_scale = torch.zeros(*batch_shape, num_components, dim, dtype=dtype) if diag else\
            torch.zeros(*batch_shape, num_components, dim, dim, dtype=dtype)
        fan_in_dims = [-1, -2] if diag else [-1, -2, -3]
        self.scale = nn.Parameter(init_log_scale / gain)  # noqa
        parametrize.register_parametrization(self, 'scale', EqualizedLR(fan_in_dims=fan_in_dims, gain=gain))
        parametrize.register_parametrization(self, "scale", ExpScaleTril(diag=diag))

        # we keep logits instead of actual weight vector to ensure valid simplex
        self.weights = nn.Parameter(torch.log(torch.ones(*batch_shape, num_components, dtype=dtype) / num_components), False)
        parametrize.register_parametrization(self, 'weights', EqualizedLR(fan_in_dims=[-1], gain=gain))
        parametrize.register_parametrization(self, "weights", nn.Softmax(-1))

    @property
    def variances(self) -> Tensor:
        return self.distribution.component_distribution.variance \
            if self.diag else self.distribution.component_distribution.covariance_matrix

    @property
    def distribution(self) -> D.MixtureSameFamily:
        # [*batch_dims, comp, dim] --> [*batch_dims, 1, comp, dim] to evaluate batched inputs
        mix = D.Categorical(self.weights.unsqueeze(-2))
        comp = D.Independent(D.Normal(self.means.unsqueeze(-3), self.scale.unsqueeze(-3)), 1) if self.diag else \
             D.MultivariateNormal(self.means.unsqueeze(-3), scale_tril=self.scale.unsqueeze(-4))
        return D.MixtureSameFamily(mix, comp)

    def log_prob_mix(self, x: Tensor) -> Tensor:
        """
        :param x: [*batch_dims, B, dim]
        """
        if self.distribution._validate_args: self.distribution._validate_sample(x)  # noqa
        x = self.distribution._pad(x)  # noqa [*batch_dims, B, 1, dim]
        log_prob_x = self.distribution.component_distribution.log_prob(x)  # [*batch_dims, B, comp]
        log_mix_prob = torch.log_softmax(self.distribution.mixture_distribution.logits, dim=-1)  # [*batch_dims, 1, comp]
        return log_prob_x + log_mix_prob  # [*batch_dims, B, comp]

    def forward(self, x: Tensor):
        # the loss is the negative log likelihood of x under the model
        return -self.distribution.log_prob(x).mean()

    @staticmethod
    @torch.enable_grad()
    def fit(gmm, x: Tensor, batch_size=100, lr=0.2, betas=(0.9, 0.999), weight_decay=1e-3, n_epochs=100) -> None:
        ddp = Accelerator()
        gmm = ddp.prepare_model(gmm)
        optim = ddp.prepare_optimizer(torch.optim.AdamW(gmm.parameters(), lr=lr, weight_decay=weight_decay, betas=betas))
        dl = ddp.prepare_data_loader(DataLoader(TensorDatasetDim(x.detach().requires_grad_(False), dim=-2),
                                                batch_size=batch_size, shuffle=True, collate_fn=partial(torch.cat, dim=-2)))
        for _ in range(n_epochs):
            for batch in dl:
                optim.zero_grad()
                nll = gmm(batch)
                nll.backward()
                optim.step()
        ddp.clear()

    def predict(self, x: Tensor, smooth: bool = False, one_hot: bool = False) -> Tensor:
        log_probs_mix = self.log_prob_mix(x)  # [*batch_dim, B, comp]
        assignments = log_probs_mix.softmax(-1) if smooth else log_probs_mix.argmin(-1)
        if not smooth and one_hot: assignments = F.one_hot(assignments, self.num_components).to(x.dtype)
        return assignments

    def assign_mean_var(self, assignments: Tensor) -> Tuple[Tensor, Tensor]:
        if self.diag:
            # [*batch_dims, 1, comp, dim] --> [*batch_dims, comp, dim]
            variances = self.variances.unsqueeze(-3)
        else:
            # [*batch_dims, 1, comp, dim, dim] --> [*batch_dims, comp, dim * dim]
            variances = self.variances.unsqueeze(-4).flatten(-2)

        # [*batch_dim, B, comp] x [*batch_dim, comp, dim] --> [*batch_dim, B, dim]
        mean = assignments @ self.means
        var = assignments @ variances
        if not self.diag:
            var = var.unflatten(-1, (self.dim, self.dim))  # [*batch_dim, B, dim * dim] --> [*batch_dim, B, dim, dim]
        return mean, var

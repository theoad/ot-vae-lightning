"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of Gaussian Mixture Model (GMM)

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.utilities import rank_zero_only
from ot_vae_lightning.utils.elr import EqualizedLR


class ExpScaleTril(nn.Module):
    def __init__(self, diag):
        super().__init__()
        self.diag = diag

    def forward(self, x: Tensor) -> Tensor:
        if self.diag: return x.exp()
        return x.tril(-1) + torch.diag_embed(x.diagonal(dim1=1, dim2=2).exp())


class GaussianMixtureModel(nn.Module):
    def __init__(self, num_components, input_dim, means_init=None, diag=False, requires_grad=True, gain=1.):
        super().__init__()
        self.num_components = num_components
        self.input_dim = input_dim
        self.diag = diag

        self.means = nn.Parameter(means_init or torch.randn(num_components, input_dim), requires_grad)
        parametrize.register_parametrization(self, 'means', EqualizedLR(gain=gain))

        # Cholesky decomposition of the covariance matrices with Exp+tril reparametrization to ensure validity
        init_log_scale = torch.zeros(num_components, input_dim) if diag else torch.zeros(num_components, input_dim, input_dim)
        self.scale = nn.Parameter(init_log_scale / gain, requires_grad)  # noqa
        parametrize.register_parametrization(self, 'scale', EqualizedLR(gain=gain))
        parametrize.register_parametrization(self, "scale", ExpScaleTril(diag=diag))

        # we keep logits instead of actual wight vector to ensure valid simplex
        self.weights = nn.Parameter(torch.log(torch.ones(num_components) / num_components), False)
        parametrize.register_parametrization(self, 'weights', EqualizedLR(gain=gain))
        parametrize.register_parametrization(self, "weights", nn.Softmax(-1))

    @property
    def variances(self) -> Tensor:
        return self.distribution.component_distribution.covariance_matrix if self.diag else self.distribution.component_distribution.variance

    @property
    def distribution(self) -> D.MixtureSameFamily:
        mix = D.Categorical(self.weights)
        comp = D.Independent(D.Normal(self.means, self.scale), 1) if self.diag else \
             D.MultivariateNormal(self.means, scale_tril=self.scale)
        return D.MixtureSameFamily(mix, comp)

    def log_prob_mix(self, x: Tensor) -> Tensor:
        if self.distribution._validate_args: self.distribution._validate_sample(x)  # noqa
        x = self.distribution._pad(x)  # noqa
        log_prob_x = self.distribution.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.distribution.mixture_distribution.logits, dim=-1)  # [B, k]
        return log_prob_x + log_mix_prob

    def forward(self, x: Tensor):
        log_probs_mix = self.log_prob_mix(x)
        assignments = torch.argmin(log_probs_mix, dim=-1)
        return assignments

    @rank_zero_only
    @torch.enable_grad()
    def fit(self, x: Tensor, batch_size=100, lr=0.2, betas=(0.9, 0.999), weight_decay=1e-3, n_epochs=100) -> None:
        # TODO: make deterministic by fixing the dataloader generator
        optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        dl = DataLoader(TensorDataset(x.detach()), batch_size=batch_size, shuffle=True)
        for _ in range(n_epochs):
            for batch in dl:
                optim.zero_grad()
                # the loss is the negative log likelihood of x under the model
                nll = -self.distribution.log_prob(batch[0]).mean()
                nll.backward()
                optim.step()

    def predict(self, x: Tensor) -> Tensor:
        return self(x)


if __name__ == "__main__":
    model = GaussianMixtureModel(100, 2, diag=False, requires_grad=True, gain=1.)

    # means = torch.Tensor([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]) * 3
    # covs = torch.Tensor([[[1., 0.], [0., 2.]], [[2., -1.], [-1., 2.]], [[1., 0.], [0., 1]], [[2., 1.], [1., 2.]]])
    means = torch.randn(100, 2) * 15
    covs = torch.randn(100, 2, 2)
    covs = covs @ covs.transpose(-1, -2)
    mix = D.Categorical(torch.ones(100, ))
    comp = D.MultivariateNormal(means, covs)
    gmm = D.MixtureSameFamily(mix, comp)

    all_x = gmm.sample((10000,))
    model.fit(all_x)
    print(model.means)
    print(model.variances)
    print(model.weights)
    # print(D.Categorical(logits=model(torch.randn(100, 2) + means[0])).sample())

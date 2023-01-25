"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a Gaussian Prior

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Optional
import torch
from torch import Tensor
from torch.types import _size, _device
from ot_vae_lightning.prior.base import Prior
from torch.distributions import Distribution, Normal
from ot_vae_lightning.utils import unsqueeze_like

__all__ = ['GaussianPrior']


class GaussianPrior(Prior):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a Gaussian Prior

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(
            self,
            loss_coeff: float = 1.,
            empirical_kl: bool = False,
            reparam_dim: int = 1,
            annealing_steps: int = 0,
            fixed_var: bool = False,
    ):
        r"""
        Initializes a vanilla Gaussian prior

        .. math::

            q(z|x) = \mathcal{N}(\mu(x), \sigma(x))

            p(z) = \mathcal{N}(0, I)

        :param loss_coeff: balancing coefficient of the prior loss
        :param empirical_kl: set to compute the KL using monte-carlo instead of closed-form
        :param reparam_dim: Re-parametrization trick is performed on the specified dimension.
        :param annealing_steps: the number of cosine annealing steps given to the prior loss to warm-up.
        :param fixed_var: if ``True``, will fix the re-parametrized distribution to have a unit variance.
        """
        super(GaussianPrior, self).__init__(loss_coeff, annealing_steps)
        self.empirical_kl = empirical_kl
        self.reparam_dim = reparam_dim
        self.fixed_var = fixed_var

    @staticmethod
    def closed_form_reverse_kl(p: Distribution, q: Distribution) -> Tensor:
        """KL(q, p), assuming q and p are Normally distributed"""
        dims = list(range(1, q.mean.dim()))
        return torch.sum(0.5 * (
            (q.mean - p.mean) ** 2 / p.variance +
            p.variance.log() - q.variance.log() +
            q.variance / p.variance - 1), dim=dims
        )

    def reparametrization(self, z: Tensor, temperature: Optional[Tensor] = None) -> Distribution:
        if self.fixed_var:
            mu, var = z, torch.ones_like(z)
            if temperature is not None:
                var = var * unsqueeze_like(temperature, var) + 1e-8
        else:
            mu, log_var = torch.chunk(z, 2, self.reparam_dim)
            var = (log_var/2).exp()
        return Normal(mu, var)

    def out_size(self, size: _size) -> _size:
        if self.fixed_var: return size
        reparam_size = list(size)
        reparam_dim = self.reparam_dim - 1 if self.reparam_dim > 0 else self.reparam_dim
        reparam_size[reparam_dim] //= 2   # re-parametrization trick
        return torch.Size(reparam_size)

    def encode(self, x: Tensor, time: Optional[Tensor] = None) -> Prior.EncodingResults:
        q = self.reparametrization(x, temperature=time)
        p = self.reparametrization(torch.zeros_like(x))
        z = q.rsample()
        loss = self.empirical_reverse_kl(p, q, z) if self.empirical_kl else self.closed_form_reverse_kl(p, q)
        artifacts = {'prior': p, 'distribution': q}
        return z, loss, artifacts

    def sample(self, shape: _size, device: _device) -> Tensor:
        return torch.randn(*shape, device=device)

    def forward(self, x: Tensor, step: int, time: Optional[Tensor] = None) -> Prior.EncodingResults:
        return super().forward(x, step, time=time)

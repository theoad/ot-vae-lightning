"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a Gaussian Prior

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""

from typing import Tuple
import torch
from torch import Tensor
from torch.types import _size
from ot_vae_lightning.prior import Prior
from torch.distributions import Distribution, Normal


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
            fixed_var: bool = False
    ):
        """
        :param loss_coeff: balancing coefficient of the prior loss
        :param empirical_kl: set to compute the KL using monte-carlo instead of closed-form
        :param reparam_dim: Re-parametrization trick is performed on the specified dimension.
        """
        super(GaussianPrior, self).__init__(loss_coeff, annealing_steps)
        self.empirical_kl = empirical_kl
        self.reparam_dim = reparam_dim
        self.fixed_var = fixed_var

    @staticmethod
    def closed_form_reverse_kl(p: Distribution, q: Distribution) -> Tensor:
        """KL(q, p), assuming q and p are Normally distributed"""
        reduce_dim = list(range(1, len(q.mean.shape)))
        return torch.sum(0.5 * (q.mean - p.mean) ** 2 / p.variance
                         + p.variance.log() - q.variance.log()
                         + q.variance / p.variance - 1, dim=reduce_dim)

    def reparametrization(self, z: Tensor) -> Distribution:
        if self.fixed_var:
            mu, var = z, torch.ones_like(z)
        else:
            mu, log_var = torch.chunk(z, 2, self.reparam_dim)
            var = (log_var/2).exp()
        return Normal(mu, var)

    def out_size(self, size: _size) -> _size:
        if self.fixed_var:
            return size
        reparam_size = list(size)
        reparam_dim = self.reparam_dim - 1 if self.reparam_dim > 0 else self.reparam_dim
        reparam_size[reparam_dim] //= 2  # re-parametrization trick
        return torch.Size(reparam_size)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.reparametrization(x)
        p = self.reparametrization(torch.zeros_like(x))
        z = q.rsample()
        if self.empirical_kl: loss = self.empirical_reverse_kl(p, q, z)
        else: loss = self.closed_form_reverse_kl(p, q)
        return z, loss

    def sample(self, shape, device) -> Tensor:
        return torch.randn(*shape, device=device)

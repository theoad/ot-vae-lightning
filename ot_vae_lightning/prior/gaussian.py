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
from ot_vae_lightning.prior import Prior
from torch.distributions import Distribution, Normal


class GaussianPrior(Prior):
    """
    Vanilla Gaussian prior. If the input is more than one dimensional,
    Re-parametrization trick is performed on the channel dimension.
    """
    def __init__(
            self,
            loss_coeff: float = 1.,
            empirical_kl: bool = False
    ):
        """
        :param loss_coeff: balancing coefficient of the prior loss
        :param empirical_kl: set to compute the KL using monte-carlo instead of closed-form
        """
        super(GaussianPrior, self).__init__(loss_coeff)
        self.empirical_kl = empirical_kl

    @staticmethod
    def closed_form_reverse_kl(
            p: Distribution,
            q: Distribution,
    ) -> Tensor:
        """KL(q, p), assuming q and p are Normally distributed"""
        reduce_dim = list(range(1, len(q.mean.shape)))
        return torch.sum(0.5 * (q.mean - p.mean) ** 2 / p.variance
                         + p.variance.log() - q.variance.log()
                         + q.variance / p.variance - 1, dim=reduce_dim)

    @staticmethod
    def reparametrization(z: Tensor) -> Distribution:
        mu, log_var = torch.chunk(z, 2, 1)
        q = Normal(mu, (log_var/2).exp())
        return q

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.reparametrization(x)
        p = self.reparametrization(torch.zeros_like(x))
        z = p.rsample()
        if self.empirical_kl: loss = self.empirical_reverse_kl(p, q, z)
        else: loss = self.closed_form_reverse_kl(p, q)
        return z, loss

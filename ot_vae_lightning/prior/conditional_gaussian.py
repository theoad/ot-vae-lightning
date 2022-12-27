"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a Conditional Gaussian Prior

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Tuple, Optional
import functools

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.types import _size
from ot_vae_lightning.prior.base import Prior
from ot_vae_lightning.prior.gaussian import GaussianPrior
from numpy import prod
from torch.distributions import Distribution, Normal
import ot_vae_lightning.utils as utils

__all__ = ['ConditionalGaussianPrior']


class ConditionalGaussianPrior(GaussianPrior):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a Conditional Gaussian Prior

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(
            self,
            dim: _size,
            num_classes: int,
            loss_coeff: float = 1.,
            empirical_kl: bool = False,
            reparam_dim: int = 1,
            annealing_steps: int = 0,
            fixed_var: bool = False,
            embedding_ema_decay: Optional[float] = None,
            eps: float = 1e-5
    ):
        r"""
        Initializes a Gaussian prior conditioned on a discrete condition (e.g. class label)

        .. math::

            q(z|x, y) = \mathcal{N}(\mu_y(x), \sigma_y^2(x) \cdot I)

            p(z|y) = \mathcal{N}(\mu_y, \sigma_{y}^2 \cdot I)

        :param dim: the dimension of the Tensor after re-parametrization.
        :param num_classes: the number of classes on which ton condition the prior
        :param loss_coeff: balancing coefficient of the prior loss
        :param empirical_kl: set to compute the KL using monte-carlo instead of closed-form
        :param reparam_dim: Re-parametrization trick is performed on the specified dimension.
        :param annealing_steps: the number of cosine annealing steps given to the prior loss to warm-up.
        :param fixed_var: if ``True``, will fix the re-parametrized distribution to have a unit variance.
        :param embedding_ema_decay: if given, will update the components according to a running exponential moving
                              wasserstein-2 barycenter of the observed samples instead of learning them with SGD.
        :param eps: A stabilization constant relevant for the EMA update.
        """
        super().__init__(loss_coeff, empirical_kl, reparam_dim, annealing_steps, fixed_var)
        self.dim = dim
        self.num_classes = num_classes
        self.decay = embedding_ema_decay
        self.eps = eps
        self.ema = functools.partial(utils.ema_inplace, decay=self.decay)

        self._mu = torch.nn.Embedding(num_classes, int(prod(dim)), _weight=-torch.rand(num_classes, int(prod(dim))))
        self._log_std = torch.nn.Embedding(num_classes, int(prod(dim)), _weight=-torch.rand(num_classes, int(prod(dim))))

        if self.decay is not None and self.decay > 0:
            self.register_buffer("_size", torch.zeros(num_classes))
            self.register_buffer("_mu_avg", torch.zeros_like(self._mu.weight))
            self.register_buffer("_log_std_avg", torch.zeros_like(self._log_std.weight))
            self._mu.requires_grad_(False)
            self._log_std.requires_grad_(False)

    def p(self, labels: Tensor) -> Distribution:
        return Normal(self._mu(labels).unflatten(1, self.dim), self._log_std(labels).unflatten(1, self.dim).exp())

    def encode(self, x: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:  # noqa arguments-differ
        p, q = self.p(labels), self.reparametrization(x)
        z = q.rsample()
        loss = self.empirical_reverse_kl(p, q, z) if self.empirical_kl else self.closed_form_reverse_kl(p, q)
        if self.decay is not None and self.decay > 0 and self.training:
            self.ema_update(q, labels)
        return z, loss

    def sample(self, shape: _size, device: torch.device, labels: Tensor) -> Tensor:  # noqa arguments-differ
        return self.p(labels).sample().to(device)

    def ema_update(self, q: Distribution, labels: Tensor) -> None:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).type(q.mean.dtype)  # [batch, num_classes]

        sizes = one_hot.sum(dim=0)
        mu_sum = one_hot.transpose(-2, -1) @ q.mean.flatten(1)  # [num_classes, batch] x [batch, dim]
        log_std_sum = one_hot.transpose(-2, -1) @ q.stddev.log().flatten(1)  # [num_classes, batch] x [batch, dim]

        # all_reduce(sizes)
        # all_reduce(mu_sum)
        # all_reduce(log_std_sum)

        self.ema(self._size, sizes)
        self.ema(self._mu_avg, mu_sum)
        self.ema(self._log_std_avg, log_std_sum)

        sizes = utils.laplace_smoothing(self._size, self.num_classes, self.eps) * self._size.sum()

        self._mu.weight.copy_(self._mu_avg.data / sizes.unsqueeze(-1))
        self._log_std.weight.copy_(self._log_std_avg.data / sizes.unsqueeze(-1))

    def forward(self, x: Tensor, step: int, labels: Tensor) -> Tuple[Tensor, Tensor]:   # noqa arguments-differ
        return Prior.forward(self, x, step, labels=labels)    # type: ignore[arg-type]

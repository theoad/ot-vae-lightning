"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of Gaussian Mixture Model (GMM)

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Union
from functools import partial

from pytorch_lightning.utilities.distributed import sync_ddp_if_available
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as D
from torch.types import _size, _dtype

from ot_vae_lightning.ot.matrix_utils import mean_cov


class GaussianModel(nn.Module):
    def __init__(self, batch_shape: _size, dim: int, dtype: _dtype, diag: bool, persistent: bool = False):
        super().__init__()
        self.batch_shape = batch_shape
        self.dim = dim
        self.diag = diag
        self.dtype = dtype
        vec_init = torch.zeros(*batch_shape, dim, dtype=dtype)
        mat_init = vec_init if diag else torch.zeros(*batch_shape, self.dim, self.dim, dtype=dtype)

        self.register_buffer("_running_sum", vec_init.clone(), persistent=persistent)
        self.register_buffer("_running_sum_cov", mat_init.clone(), persistent=persistent)
        self.register_buffer("_n_obs", torch.zeros([], dtype=torch.long), persistent=persistent)

        self._reduce = partial(sync_ddp_if_available, reduce_op="sum")
        self.mean, self.cov = None, None

    def reset(self):
        self._running_sum.zero_()
        self._running_sum_cov.zero_()
        self._n_obs.zero_()

    @property
    def distribution(self) -> Union[D.Independent, D.MultivariateNormal]:
        return D.Independent(D.Normal(self.mean, self.cov), 1) if self.diag else\
            D.MultivariateNormal(self.mean, covariance_matrix=self.cov)

    @property
    def variances(self) -> Tensor:
        return self.distribution.variance if self.diag else self.distribution.covariance_matrix

    def forward(self, samples: Tensor) -> None:
        """
        :param samples: samples of the distribution to fit. [*batch_shape, B, dim]
        """
        if samples.shape[:-2] != self.batch_shape:
            ValueError(f"`source_samples` is expected to have leading dimensions equaled to {self.batch_shape}")
        if samples.size(-1) != self.dim:
            ValueError(f"`source_samples` is expected to have dimensionality equaled to {self.dim}")
        samples = samples.to(self.dtype)
        self._n_obs += samples.size(-2)
        self._running_sum += samples.sum(-2)
        if self.diag: self._running_sum_cov += (samples ** 2).sum(-2)
        else: self._running_sum_cov += (samples.unsqueeze(-1) @ samples.unsqueeze(-2)).sum(-3)

    def fit(self):
        self.mean, self.cov = mean_cov(
            self._reduce(self._running_sum),
            self._reduce(self._running_sum_cov),
            self._reduce(self._n_obs),
            diag=self.diag
        )

"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a pure virtual distributions model class

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple, Literal
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.types import _device, _dtype
import torch.distributions as D
import torch.nn.functional as F

import ot_vae_lightning.utils as utils

__all__ = ['DistributionModel', 'MixtureMixin']


class DistributionModel(nn.Module, utils.DDPMixin, ABC):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a pure vitrual distributions model class

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved
    
    .. warning:: Work in progress. This implementation is still being verified.
    
    .. _TheoA: https://github.com/theoad
    """
    Distribution = None

    def __init__(
            self,
            *size: int,
            reduce_on_update: bool = True,
            update_decay: Optional[float] = None,
            update_with_autograd: bool = False,
            device: Optional[_device] = None,
            dtype: Optional[_dtype] = None,
            **ddp_kwargs
    ):
        nn.Module.__init__(self)
        utils.DDPMixin.__init__(self, **ddp_kwargs)
        self.leading_shape = torch.Size(size[:-1])
        self.dim = size[-1]
        self.reduce_on_update = reduce_on_update
        self.decay = update_decay
        self.ema_update = partial(utils.ema, decay=update_decay)
        self.register_buffer("vec_init", torch.randn(*self.vec_shape, dtype=dtype, device=device))
        self.register_buffer("mat_init", torch.randn(*self.vec_shape, self.dim, dtype=dtype, device=device))
        self.update_with_autograd = update_with_autograd

    @property
    def vec_shape(self):
        return *self.leading_shape, self.dim

    def _broadcastable(self, shape):
        return torch.broadcast_shapes(shape, self.leading_shape) == self.leading_shape

    def _validate_samples(self, samples: Tensor) -> None:
        """
        Check the samples have the expected shape

        :param samples: tensor with shape [*leading_shape, batch, dim]
        """
        if not self._broadcastable(samples.shape[:-2]):
            ValueError(f"`samples` leading dimensions are expected to broadcast to `self.leading_shape`={self.leading_shape}")
        if samples.size(-1) != self.dim:
            ValueError(f"`samples` are expected to have dimensionality equaled to `self.dim`={self.dim}")

    def _fit_warn(self):
        if self.update_with_autograd:
            self.warn("`self.update_with_autograd` is True. The parameters should be updated with autograd."
                      "`fit` will override the nn.Parameters updated by training with values"
                      " computed from running internal states.")

    def _update_warn(self):
        if self.update_with_autograd:
            self.warn("`self.update_with_autograd` is True. The parameters should be updated with autograd."
                      "`update` cannot access running internal states since they were not created in `__init__`")

    @abstractmethod
    def reset(self) -> None:
        """reset internal model states"""

    @property
    def distribution(self) -> D.Distribution:
        """
        instantiates the distribution implemented by the model

        :return: torch distribution with `batch_shape`=leading_shape and `even_shape`=(dim,)
        """
        raise NotImplementedError()

    @property
    def batched_distribution(self) -> D.Distribution:
        """
        instantiates the distribution implemented by the model
        with an additional batch dimension (to evaluate probability on batches)

        :return: torch distribution with `batch_shape`=(*leading_shape, 1) and `even_shape`=(dim,)
        """
        raise NotImplementedError()

    @property
    def variances(self) -> Tensor:
        """returns the covariance/variance of the distribution implemented by the model"""
        raise NotImplementedError()

    def forward(self, samples: Tensor) -> Any:
        """
        Fits the model to the evidence if `requires_grad` disabled and returns the model
        predictions on the samples.

        :param samples: realisation of the distribution to fit. [*leading_shape, batch, dim]
        """
        self._validate_samples(samples)
        if self.training and not self.update_with_autograd:
            self.update(samples)
        return self.predict(samples)

    @abstractmethod
    def update(self, samples: Tensor) -> None:
        """
        updates the model internal states on-the-fly to fit the model to the given samples

        :param samples: realisation of the distribution to fit. [*leading_shape, batch, dim]
        """

    @abstractmethod
    def fit(self, samples: Optional[Tensor] = None) -> None:
        """
        fits the model to the given samples

        :param samples: realisation of the distribution to fit. [*leading_shape, batch, dim]
        """

    @abstractmethod
    def predict(self, samples: Tensor) -> Any:
        """
        Varies per inheritance.
        E.g. in the case of a GMM it returns assignments.
        When `requires_grad` is enabled, the function can serve as a proxy to update
        the internal parameters using autograd.
        """

    @abstractmethod
    def w2(self, other: Distribution) -> Tensor:
        """returns the w2 distance to the `other` distribution"""

    def extra_repr(self) -> str:
        return f"""leading_dim={tuple(self.leading_shape)}, dim={self.dim}, decay={self.decay}, update_with_autograd={self.update_with_autograd}"""


class MixtureMixin(ABC):
    Mode = Literal['mean', 'sample', 'argmax', 'gumbel-softmax', 'gumbel-hardmax']

    def __init__(
            self,
            *leading_shape,
            n_components: int,
            metric: Literal['cosine', 'euclidean'] = 'euclidean',
            p: float = 2.,
            topk: Optional[int] = None,
            temperature: float = 1.,
            training_mode: Mode = 'argmax',
            inference_mode: Mode = 'argmax',
            kmeans_iter: int = 100,
            laplace_eps: Optional[float] = 1e-5,
    ):
        super().__init__()
        self.n_components = n_components
        self.metric = metric
        self.topk = topk
        self.temperature = temperature
        self.training_mode = training_mode
        self.inference_mode = inference_mode
        self.kmeans_iter = kmeans_iter
        self.p = p

        weight_init = torch.ones(*leading_shape, n_components)
        weight_init /= weight_init.sum(-1, keepdim=True)
        self._weight_init = weight_init

        self.laplace_smoothing = partial(utils.laplace_smoothing, n_categories=n_components, eps=laplace_eps)

    @abstractmethod
    def energy(self, samples: Tensor) -> Tensor:
        """
        Computes the similarity of each sample w.r.t. the mixture components

        :param samples: tensor of shape [*leading_shape, batch, dim]
        :return: energy matrix of shape [*leading_shape, batch, n_comp]
        """

    def assign(self, samples: Tensor) -> Tuple[Tensor, Tensor, D.Categorical]:
        """
        Assigns each sample to a component

        :param samples: tensor of shape [*leading_shape, batch, dim]
        :return:
            - weight assignments with shape [*leading_shape, batch, n_comp]
            - assignment indices (either most probable or sampled) with shape [*leading_shape, batch]
            - assignment distribution (either one-hot or smoothed) with shape [*leading_shape, batch, n_comp]
        """
        energy = self.energy(samples)
        if self.topk is not None and self.topk > 0:
            val, idx = torch.topk(energy, self.topk, dim=-1)
            energy = -torch.ones_like(energy) * float('inf')
            energy.scatter_(-1, idx, val)

        weights = torch.softmax(energy / self.temperature, dim=-1)
        distribution = D.Categorical(weights)
        indices = distribution.sample()

        training = self.training if hasattr(self, 'training') else False
        mode = self.training_mode if training else self.inference_mode
        if mode == 'mean' or self.topk == 1:
            pass
        elif mode == 'sample':
            weights = F.one_hot(indices, energy.size(-1)).type_as(weights)
        elif mode == 'argmax':
            weights = F.one_hot(weights.argmax(-1), energy.size(-1)).type_as(weights)
        elif 'gumbel' in mode:
            weights = F.gumbel_softmax(energy, tau=self.temperature, hard='hard' in mode, dim=-1)
        else:
            raise NotImplementedError(f"`mode` must be 'sample', 'mean', 'argmax', 'gumbel' or 'hard-gumbel'. Got `mode`={mode}")

        return weights, indices, distribution

    def kmean_iteration(self, samples: Tensor) -> Tuple[Tensor, ...]:
        """
        Sums all the samples belonging to each component with their relative weight

        :param samples: tensor of shape [*leading_shape, batch, dim]
        :return:
            - samples weighted sum per component with shape [*leading_shape, n_comp, dim]
            - weight sum per component [*leading_shape, n_comp]
        """
        weights, _, _ = self.assign(samples)  # [*, b, n_comp]
        weights_sum = weights.sum(-2)  # [*, n_comp]
        samples_weighted_sum = weights.transpose(-1, -2) @ samples  # [*, n_comp, b] x [*, b, d] --> [*, n_comp, d]
        return weights_sum, samples_weighted_sum

    @abstractmethod
    def _update_parameters(self, *kmeans_iter_res: Tensor) -> None:
        """updates the internal parameters of the model"""

    @abstractmethod
    def _update_buffers(self, *kmeans_iter_res: Tensor, decay=False):
        """updates the internal buffers of the model"""

    def extra_repr(self) -> str:
        return f"""num_components={self.n_components}, metric={self.metric}, topk={self.topk}, p={self.p},
temperature={self.temperature}, training_mode={self.training_mode}, inference_mode={self.inference_mode}"""

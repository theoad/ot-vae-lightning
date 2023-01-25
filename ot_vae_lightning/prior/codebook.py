from typing import Optional, Literal
from math import log
from numpy import prod
from functools import partial
from math import cos, pi

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.types import _size, _device
import torch.distributions as D

from ot_vae_lightning.prior.base import Prior
from ot_vae_lightning.ot.distribution_models.codebook_model import CodebookModel
import ot_vae_lightning.utils as utils

__all__ = ['CodebookPrior']


class CodebookPrior(Prior):
    def __init__(self,
                 latent_size: _size,
                 embed_dims: _size,
                 loss: Optional[Literal['kl', 'first_kl', 'l2']] = None,
                 temperature_annealing: Optional[int] = None,
                 loss_coeff: float = 1.,
                 annealing_steps: int = 0,
                 **codebook_kwargs
                 ):
        r"""
        :param latent_size: the size of the Tensors to embed
        :param embed_dims: The dimension on which the linear re-weighting of the codebook atoms is applied
         If (1, 2, 3), will embed the input vectors as a whole
         If (1,), will embed each needle (pixel) individually - similar to VQVAE
         if (2, 3), will embed each channel individually
        """
        super().__init__(loss_coeff=loss_coeff, annealing_steps=annealing_steps)
        all_dims = list(range(1, len(latent_size) + 1))
        if not set(embed_dims).issubset(all_dims):
            raise ValueError(f"""
                    Given `latent_size`={latent_size}. The inputs will have {len(latent_size) + 1} dimensions
                    (with the batch dimensions). Therefore `embed_dims` must be a subset of {self.all_dims}.
                    """)

        self.size = latent_size
        self.embed_dims = embed_dims
        self.batch_dims = torch.Size(list(set(all_dims).difference(set(self.embed_dims))))
        self.event_shape = torch.Size([latent_size[i - 1] for i in self.embed_dims])
        self.batch_shape = torch.Size([latent_size[i - 1] for i in self.batch_dims])
        self.dimensionality = prod(self.event_shape)
        self.permute_and_flatten = partial(
            utils.permute_and_flatten,
            permute_dims=self.embed_dims,
            batch_first=False,
            flatten_batch=False
        )
        self.unflatten_and_unpermute = partial(
            utils.unflatten_and_unpermute,
            orig_shape=torch.Size([-1, *self.size]),
            permute_dims=self.embed_dims,
            batch_first=False,
            flatten_batch=False
        )
        self.loss = loss
        self.codebook_model = CodebookModel(1, self.dimensionality, **codebook_kwargs)
        self.commitment_cost = 0. if self.codebook_model.training_mode in ['sample', 'argmax'] else 0.1
        self.temperature_annealing = temperature_annealing
        self.original_temperature = self.codebook_model.temperature

    @property
    def num_embeddings(self):
        return self.codebook_model.n_components

    def out_size(self, size: _size) -> _size:
        return size

    def _compute_loss(self, x: Tensor, encodings: Tensor, dist: D.Distribution) -> Tensor:
        if self.loss is None: prior_loss = torch.zeros(x.size(-2), device=x.device).type_as(x)
        elif self.loss.lower() == 'l2': prior_loss = F.mse_loss(x, encodings.detach(), reduction='none').mean(-1).sum(0)
        elif self.loss.lower() == 'kl': prior_loss = (log(self.num_embeddings)-dist.entropy()).sum(0)
        elif self.loss.lower() == 'first_kl': prior_loss = (log(self.num_embeddings)-dist.entropy())[0]
        else: raise NotImplementedError(f"loss must be 'l2', 'kl' or 'first_kl'. Given: {self.loss}")

        if self.commitment_cost > 0:
            embedding_loss = F.mse_loss(encodings, x.detach(), reduction='none').mean(-1).sum(0)
            prior_loss = prior_loss + self.commitment_cost * embedding_loss
        return prior_loss

    def encode(self, x: Tensor) -> Prior.EncodingResults:
        x = self.permute_and_flatten(x)  # [prod(self.batch_shape), batch, dimensionality]

        encodings, indices, dist = self.codebook_model(x)  # [prod(self.batch_shape), batch, ...]
        prior_loss = self._compute_loss(x, encodings, dist)

        if self.codebook_model.training_mode in ['sample', 'argmax']:
            encodings = x + (encodings - x).detach()   # straight-through estimator

        encodings = self.unflatten_and_unpermute(encodings)
        # TODO: This is a hack. Need to unflatten and unpermute properly
        dist.probs = dist.probs.transpose(0, 1)
        indices = indices.transpose(0, 1)

        artifacts = {'distribution': dist, 'indices': indices}

        return encodings, prior_loss, artifacts

    def sample(self, shape: _size, device: _device, mode: str = 'sample') -> Tensor:
        x = torch.randn(*shape)
        ind_shape = self.permute_and_flatten(x).shape[:-1]
        encodings = self.codebook_model.distribution.sample(ind_shape).squeeze(-2)
        encodings = self.unflatten_and_unpermute(encodings)
        return encodings

    def forward(self, x: Tensor, step: int, **kwargs) -> Prior.EncodingResults:
        if self.temperature_annealing is not None and self.training:
            self.codebook_model.temperature = self.original_temperature * 0.5 * cos(pi * step / self.temperature_annealing) + 0.5
        return super().forward(x, step, **kwargs)

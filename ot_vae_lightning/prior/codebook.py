from typing import Optional, Tuple
from itertools import accumulate
from operator import mul
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from torch.types import _size

from ot_vae_lightning.prior import Prior
import ot_vae_lightning.utils as utils
from copy import deepcopy


class CodebookPrior(Prior):
    def __init__(self,
                 num_embeddings: int,
                 latent_size: _size,
                 embed_dims: _size = (1, 2, 3),
                 similarity_metric: str = 'cosine',
                 separate_key_values: bool = False,
                 loss_coeff: float = 1.,
                 annealing_steps: int = 0,
                 ):
        r"""
        :param num_embeddings: the number of atoms in the codebook
        :param latent_size: the size of the Tensors to embed
        :param embed_dims: The dimension on which the linear re-weighting of the codebook atoms is applied
         If (1, 2, 3), will embed the input vectors as a whole
         If (1,), will embed each needle (pixel) individually - similar to VQVAE
         if (2, 3), will embed each channel individually
        :param separate_key_values: will use two codebooks: one for computing similarities and one for re-weighting
        :param loss_coeff: balancing coefficient of the prior loss
        :param annealing_steps: the number of cosine annealing steps given to the prior loss to warm-up.
        """
        super().__init__(loss_coeff=loss_coeff, annealing_steps=annealing_steps)
        self.all_dims = list(range(1, len(latent_size) + 1))
        if not set(embed_dims).issubset(self.all_dims):
            raise ValueError(f"""
                    Given `latent_size`={latent_size}. The inputs will have {len(latent_size) + 1} dimensions
                    (with the batch dimensions). Therefore `embed_dims` must be a subset of {self.all_dims}.
                    """)

        if similarity_metric.lower() not in ['cosine', 'l2']:
            raise NotImplementedError(f"Supported `similarity_metric` are: 'cosine', 'l2'.")

        self.similarity_metric = similarity_metric.lower()
        self.num_embeddings = num_embeddings
        self.latent_size = latent_size
        self.embed_dims = embed_dims
        self.embedding_dim = list(accumulate([latent_size[i - 1] for i in self.embed_dims], mul))[-1]

        self.values = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.keys = deepcopy(self.values.weight) if separate_key_values else self.values.weight

    def out_size(self, size: _size) -> _size:
        return size

    def weights(self, input: Tensor, temperature: float, topk: Optional[int]) -> torch.Tensor:
        """
        Computes the quantization distribution for each embedding in `input`
        :param input: Tensor of shape [*, embedding_dim] to quantize
        :param temperature: distribution temperature.
        :param topk: If not None, will only zero the weights of embeddings with the smallest similarity and only keep
                     the top-k embeddings.
        :return: Categorical distributions weights of shape [*, num_embeddings | topk]
        """
        assert input.size(-1) == self.embedding_dim

        norm_x = torch.sum(input ** 2, dim=-1, keepdim=True)  # [..., 1]
        norm_k = torch.sum(self.keys ** 2, dim=-1).unsqueeze(0)  # [1, n_embeddings, 1]
        dot_product = input @ self.keys.transpose(-2, -1)  # [..., n_embeddings]

        if self.similarity_metric == 'l2': energy = -(norm_k + norm_x - 2 * dot_product)
        elif self.similarity_metric == 'cosine': energy = dot_product / norm_x / norm_k
        else: raise NotImplementedError(f"Supported `similarity_metric` are: 'cosine', 'l2'.")

        if topk is not None:
            val, idx = torch.topk(energy, topk, dim=-1)
            energy = -torch.ones_like(energy) * float('inf')
            energy.scatter_(-1, idx, val)  # [..., topk]

        weights = torch.softmax(energy / temperature, dim=-1)  # [..., topk]
        return weights

    def encode(
            self,
            x: Tensor,
            temperature: float = 1.,
            topk: Optional[int] = None,
            mode: str = 'smooth',  # 'mean', 'sample', 'argmax'
            loss: Optional[str] = None,  # 'kl', 'first_kl', 'l2'
            commitment_cost: float = 0,
            permute: bool = True,
            return_distributions: bool = False,
            return_indices: bool = False
    ) -> Tuple[Tensor, Tensor]:
        orig_shape = x.shape
        if permute: x = utils.permute_and_flatten(x, self.embed_dims)  # [batch, n_encodings, embedding_dim]

        weights = self.weights(x, temperature, topk)    # [batch, n_encodings, num_embeddings]
        dist = Categorical(weights)
        indices = dist.sample()

        if mode == 'mean': pass
        elif mode == 'sample': weights = F.one_hot(indices, self.num_embeddings).type(weights.dtype)
        elif mode == 'argmax': weights = F.one_hot(weights.argmax(-1), self.num_embeddings).type(weights.dtype)
        else: raise NotImplementedError(f"mode must be 'sample', 'mean' or 'argmax'. Given: {mode}")

        encodings = weights @ self.values.weight  # [batch, n_encodings, embedding_dim]

        # Loss
        if loss is None: prior_loss = torch.zeros(x.size(0), device=x.device).type_as(x)
        elif loss.lower() == 'l2': prior_loss = F.mse_loss(x, encodings.detach(), reduction='none').mean(-1).sum(-1)
        elif loss.lower() == 'kl': prior_loss = (log(self.num_embeddings)-dist.entropy()).sum(dim=-1)
        elif loss.lower() == 'first_kl': prior_loss = (log(self.num_embeddings)-dist.entropy())[..., 0]
        else: raise NotImplementedError(f"loss must be 'l2', 'kl' or 'first_kl'. Given: {loss}")

        if commitment_cost > 0:
            embedding_loss = F.mse_loss(encodings, x.detach(), reduction='none').mean(-1).sum(-1)
            prior_loss = prior_loss + commitment_cost * embedding_loss

        if mode == 'sample': encodings = x + (encodings - x).detach()   # straight-through estimator
        if permute: encodings = utils.unflatten_and_unpermute(encodings, orig_shape, self.embed_dims)

        if not return_distributions and not return_indices: return encodings, prior_loss

        artifacts = {'loss': prior_loss}
        if return_distributions: artifacts['distributions'] = dist
        if return_indices: artifacts['indices'] = indices

        return encodings, artifacts     # type: ignore[return-type]

    def sample(self, shape, device, mode: str = 'sample'):
        x = torch.randn(*shape)
        ind_shape = utils.permute_and_flatten(x, self.embed_dims).shape[:-1]
        if mode == 'mean':
            weights = torch.ones(*ind_shape, self.num_embeddings, device=device) / self.num_embeddings
            weights = weights.to(dtype=self.values.dtype)
        elif mode == 'sample':
            embed_ind = torch.randint(high=self.num_embeddings, size=ind_shape, device=device)
            weights = F.one_hot(embed_ind, num_classes=self.num_embeddings).to(dtype=self.values.dtype)
        else: raise NotImplementedError(f"mode must be either 'sample' or 'mean'. Given: {mode}")
        encodings = weights @ self.values.weight
        return utils.unflatten_and_unpermute(encodings, shape, self.embed_dims)

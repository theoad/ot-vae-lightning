"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a modular ViT

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

Inspired from: `lucidrains' <https://github.com/lucidrains/vit-pytorch>`_ implementation

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Union, Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
from vit_pytorch.vit import pair, Transformer


class ViT(nn.Module):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a modular ViT

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    Inspired from: `lucidrains' <https://github.com/lucidrains/vit-pytorch>`_ implementation

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]],
            patch_size: Union[int, Tuple[int, int]],
            dim: int,
            depth: int = 6,
            heads: int = 8,
            mlp_dim: int = 2048,
            channels: int = 3,
            dim_head: int = 64,
            dropout: float = 0.1,
            emb_dropout: float = 0.,
            n_embed_tokens: Optional[int] = 1,
            n_input_tokens: Optional[int] = None,
            n_output_tokens: Optional[int] = None,
            patch_to_embed: bool = True,
            embed_to_patch: bool = False
    ):
        """
        Initialize a classic ViT, generalized to support both encoding and decoding.

        :param image_size: the size of the image to encode/decode.
        :param patch_size: the patch size used for pre-processing the image to token embeddings.
        :param dim: the number of expected features in the encoder/decoder inputs (default=512).
        :param depth: the number of sub-encoder-layers in the transformer (default=6).
        :param heads: the number of heads in the multiheadattention models (default=8).
        :param mlp_dim: the dimension of the feedforward network model (default=2048).
        :param channels: the number of channels in the image to encode/decode (default=3).
        :param dim_head: the dimension of each head in the multiheaded self attention (default=64).
        :param dropout: the dropout value (default=0.1).
        :param emb_dropout: the embedding dropout value (default=0.).
        :param n_embed_tokens: the number of additional tokens (like cls token) to add to the input (default=1).
        :param n_input_tokens: the number of input tokens - relevant when decoding -. If ``None``, will be set to the
                               number of patches (default=None).
        :param n_output_tokens: the number of tokens outputted by the network. The additional embedding tokens are
                                returned first. If ``None``, will be set to the total number of tokens (default=None).
        :param patch_to_embed: If ``True``, maps the image patches to the token space via a linear layer (classic ViT).
        :param embed_to_patch: If ``True``, maps the output tokens to the patch space via a linear layer.
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        if image_height % patch_height or image_width % patch_width:
            raise ValueError('Image dimensions must be divisible by the patch size.')

        n_patch_h, n_patch_w = (image_height // patch_height), (image_width // patch_width)
        num_patches, patch_dim = n_patch_h * n_patch_w, channels * patch_height * patch_width
        n_embed_tokens = num_patches if n_embed_tokens is None else n_embed_tokens
        n_input_tokens = num_patches if n_input_tokens is None else n_input_tokens
        n_output_tokens = n_embed_tokens if n_output_tokens is None else n_output_tokens

        self.patch_to_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        ) if patch_to_embed else nn.Identity()

        self.embed_to_patch = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=n_patch_h, p1=patch_height, p2=patch_width),
        ) if embed_to_patch else nn.Identity()

        self.embed_token = nn.Parameter(torch.randn(1, n_embed_tokens, dim)) if n_embed_tokens > 0 else None
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_input_tokens + n_embed_tokens, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.n_output_tokens = n_output_tokens
        self.out_size = torch.Size([n_output_tokens, dim]) if not embed_to_patch else \
            torch.Size([channels, image_height, image_width])

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_to_embed(x)

        embed_token = repeat(self.embed_token, '1 n d -> b n d', b=x.size(0))
        x = torch.cat((embed_token, x), dim=1)
        x += self.pos_embedding
        x = self.emb_dropout(x)
        x = self.transformer(x)
        x = x[:, :self.n_output_tokens]
        x = self.embed_to_patch(x)
        return x

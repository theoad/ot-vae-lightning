"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a modular ViT

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

Inspired from: `lucidrains' <https://github.com/lucidrains/vit-pytorch>`_ implementation

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Union, Tuple, Optional, Sequence

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
            n_embed_tokens: Optional[Union[int, str]] = 1,
            n_input_tokens: Optional[Union[int, str]] = None,
            output_tokens: Union[str, Sequence[str]] = 'embed',  # 'embed', 'input', 'class'
            patch_to_embed: bool = True,
            embed_to_patch: bool = False,
            num_classes: Optional[int] = None,
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
        :param output_tokens: the type of tokens outputted by the network.
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
        n_class_tokens = int(num_classes is not None)
        total_num_tokens = n_embed_tokens + n_class_tokens + n_input_tokens
        token_indices = {
            'embed': list(range(n_embed_tokens)),
            'class': list(range(n_embed_tokens, n_embed_tokens + n_class_tokens)),
            'input': list(range(n_embed_tokens + n_class_tokens, total_num_tokens))
        }
        output_tokens = [output_tokens] if isinstance(output_tokens, str) else output_tokens
        self.output_tokens_indices = []
        for token_type in token_indices.keys():
            if token_type in output_tokens:
                self.output_tokens_indices += token_indices[token_type]

        self.patch_to_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        ) if patch_to_embed else nn.Identity()

        self.embed_to_patch = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=n_patch_h, p1=patch_height, p2=patch_width),
        ) if embed_to_patch else nn.Identity()

        self.embed_token = nn.Parameter(torch.randn(1, n_embed_tokens, dim)) if n_embed_tokens > 0 else None
        self.class_token = nn.Embedding(num_classes, dim) if n_class_tokens > 0 else None
        self.pos_embedding = nn.Parameter(torch.randn(1, total_num_tokens, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        if embed_to_patch:
            self.out_size = torch.Size([channels, image_height, image_width])
            assert len(self.output_tokens_indices) == num_patches, """flow assertion error: expected the number of 
            returned tokens to match the number of patches when `embed_to_patch` is on."""
        else:
            self.out_size = torch.Size([len(self.output_tokens_indices), dim])

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self.patch_to_embed(x)

        if self.embed_token is not None:
            embed_token = repeat(self.embed_token, '1 n d -> b n d', b=x.size(0))
            x = torch.cat((embed_token, x), dim=1)

        if self.class_token is not None and y is not None:
            x = torch.cat((x, self.class_token(y).unsqueeze(1)), dim=1)
            x += self.pos_embedding

        x = self.emb_dropout(x)
        x = self.transformer(x)
        x = x[:, self.output_tokens_indices]
        x = self.embed_to_patch(x)
        return x

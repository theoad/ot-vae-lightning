"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a modular ViT

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

Inspired from: `lucidrains' <https://github.com/lucidrains/vit-pytorch>`_ implementation

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import warnings
from typing import Union, Tuple, Optional, Sequence

import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
from ot_vae_lightning.networks.utils import GaussianFourierProjection


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_length: int, d_model: int, dropout: float, batch_first: bool, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        self.position_embeddings = nn.Embedding(max_length, d_model, device=device, dtype=dtype)
        self.LayerNorm = torch.nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        if input.size(-1) != self.d_model:
            raise RuntimeError("the feature number of `input` must be equal to d_model")

        is_batched = input.dim() == 3
        seq_dim, batch_dim = int(is_batched and self.batch_first), int(not self.batch_first)
        seq_length = input.shape[seq_dim]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=input.device)
        position_embeds = self.position_embeddings(position_ids)
        if is_batched: position_embeds = position_embeds.unsqueeze(batch_dim)

        embeddings = input + position_embeds  # broadcast if is_batched == True
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ViT(nn.Module):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a modular ViT

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]],
            dim: int,
            patch_size: Optional[Union[int, Tuple[int, int]]] = None,
            depth: int = 6,
            heads: int = 8,
            mlp_dim: Optional[int] = None,
            channels: int = 3,
            dropout: float = 0.1,
            emb_dropout: float = 0.,
            n_embed_tokens: Optional[Union[int, str]] = 1,
            n_input_tokens: Optional[Union[int, str]] = None,
            output_tokens: Union[str, Sequence[str]] = 'embed',  # 'embed', 'input', 'class'
            patch_to_embed: bool = True,
            embed_to_patch: bool = False,
            num_classes: Optional[int] = None,
            time_dependant: bool = False,
            causal_mask: bool = False
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
        self.dim = dim
        self.causal_mask = causal_mask

        image_height, image_width = pair(image_size)

        mlp_dim = mlp_dim or dim * 4

        if patch_size is None:
            patch_size = min(image_height//4, 16), min(image_width//4, 16)

        patch_height, patch_width = pair(patch_size)

        if image_height % patch_height or image_width % patch_width:
            raise ValueError('Image dimensions must be divisible by the patch size.')

        n_patch_h, n_patch_w = (image_height // patch_height), (image_width // patch_width)
        self.num_patches, self.patch_dim = n_patch_h * n_patch_w, channels * patch_height * patch_width

        self.n_tokens = {
            'input': self.num_patches if n_input_tokens is None else n_input_tokens,
            'embed': self.num_patches if n_embed_tokens is None else n_embed_tokens,
            'class': int(num_classes is not None),
            'time': int(time_dependant)
        }

        self.total_num_tokens = sum(self.n_tokens.values())

        curr_idx = 0
        self.token_indices = {}
        for token_type, num_tokens in self.n_tokens.items():
            self.token_indices[token_type] = list(range(curr_idx, curr_idx + num_tokens))
            curr_idx += num_tokens

        if isinstance(output_tokens, str):
            output_tokens = [output_tokens]
        if not all(token_type in self.token_indices.keys() for token_type in output_tokens):
            raise ValueError(f"`output_tokens` must contain only keys within {self.token_indices.keys()}")
        self.output_tokens_indices = []
        for token in output_tokens:
            self.output_tokens_indices += self.token_indices[token]

        self.patch_to_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, dim),
        ) if patch_to_embed else nn.Identity()

        self.embed_to_patch = nn.Sequential(
            nn.Linear(dim, self.patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=n_patch_h, p1=patch_height, p2=patch_width),
        ) if embed_to_patch else nn.Identity()

        self.embed_token = nn.Parameter(torch.randn(1, self.n_tokens['embed'], dim)) if self.n_tokens['embed'] > 0 else None
        self.class_token = nn.Embedding(num_classes, dim) if self.n_tokens['class'] > 0 else None
        self.time_token = GaussianFourierProjection(dim, trainable=True) if self.n_tokens['time'] > 0 else None
        self.positional_embed = PositionalEmbedding(self.total_num_tokens, dim, emb_dropout, True)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout, batch_first=True),
            num_layers=depth
        )

        if embed_to_patch:
            self.out_size = torch.Size([channels, image_height, image_width])
            # assert len(self.output_tokens_indices) == self.num_patches, """flow assertion error: expected the number
            # of returned tokens to match the number of patches when `embed_to_patch` is on."""
        else:
            self.out_size = torch.Size([len(self.output_tokens_indices), dim])

    def _add_class_token(self, x: Tensor, labels: Tensor) -> Tensor:
        if labels is not None and self.class_token is None:
            warnings.warn("""
            given conditional argument `labels` but `self.class_token` is None.
            To enable class-conditioned ViT, use `vit = ViT(num_classes=...).`
            """)
        if self.class_token is not None and labels is None:
            raise ValueError("`num_classes` specified but `labels` is None. Can't infer the class token.")
        if self.class_token is not None:
            assert labels is not None, "Flow assertion error. Expected `labels` to not be None"
            x = torch.cat((x, self.class_token(labels).unsqueeze(1)), dim=1)
        return x

    def _add_time_token(self, x: Tensor, time: Tensor) -> Tensor:
        if time is not None and self.time_token is None:
            warnings.warn("""
            given conditional argument `time` but `self.time_token` is None.
            To enable time-conditioned ViT, use `vit = ViT(time_dependant=True).`
            """)
        if self.time_token is not None and time is None:
            raise ValueError("`time_dependant` specified but `time` is None. Can't infer the time token.")
        if self.time_token is not None:
            assert time is not None, "Flow assertion error. Expected `time` to not be None"
            x = torch.cat((x, self.time_token(time).unsqueeze(1)), dim=1)
        return x

    def _add_embed_token(self, x: Tensor) -> Tensor:
        if self.embed_token is not None:
            embed_token = repeat(self.embed_token, '1 n d -> b n d', b=x.size(0))
            x = torch.cat((x, embed_token), dim=1)
        return x

    def _causal_mask(self, x: Tensor) -> Optional[Tensor]:
        # TODO: should mask only input indices
        if not self.causal_mask: return None
        return nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)

    def forward(self, x: Tensor, labels: Optional[Tensor] = None, time: Optional[Tensor] = None) -> Tensor:
        x = self.patch_to_embed(x)

        x = self._add_embed_token(x)
        x = self._add_class_token(x, labels)
        x = self._add_time_token(x, time)

        x = self.positional_embed(x)
        mask = self._causal_mask(x)

        x = self.transformer(x, mask=mask)
        x = x[:, self.output_tokens_indices]

        if not isinstance(self.embed_to_patch, nn.Identity): x = x[:, -self.num_patches:]
        x = self.embed_to_patch(x)
        return x


class AutoRegressive(ViT):
    def __init__(self, vocab_size: int, **vit_kwargs):
        super().__init__(**vit_kwargs)
        self.vocab_embed = nn.Embedding(vocab_size, self.dim)
        self.head = nn.Linear(self.dim, vocab_size)

    def forward(self, x: Tensor, labels: Optional[Tensor] = None, time: Optional[Tensor] = None) -> Tensor:
        embeds = self.vocab_embed(x)
        hs = super().forward(embeds, labels, time)
        logits = self.head(hs)
        return logits

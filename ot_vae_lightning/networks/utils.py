from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import ot_vae_lightning.utils as utils


class FilterSequential(nn.Sequential):
    """
    A sequential module that filters kwargs arguments to adapt to the forward signature
    of the modules it contains.
    """
    def forward(self, x, **kwargs):
        for layer in self:
            with utils.FilterKwargs(layer, arg_keys=list(kwargs.keys())) as l:
                x = l(x, **kwargs)
        return x


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, dim: int, out_dim: Optional[int] = None, scale: float = 1., trainable: bool = True):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.weight = nn.Parameter(self._init_tensor, requires_grad=trainable)
        self.proj = nn.Linear(dim, out_dim) if out_dim is not None else nn.Identity()

    @property
    def _init_tensor(self):
        return torch.randn(1, self.dim // 2) * self.scale

    def forward(self, input):
        if input.dim() != 1:
            raise ValueError("`input` is expected to be 1-dimensional")
        if (input < 0).any() or (input > 1).any():
            raise ValueError("`input` is expected to contain floats in the range [0,1]")

        x_proj = input.unsqueeze(-1) * self.weight * 2 * np.pi
        return self.proj(torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1))


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention group-wise and splits in a different order.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x G x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C * G) x T] tensor after attention.
        """
        if qkv.dim() == 3: qkv = qkv.unsqueeze(1)
        bs, groups, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0, f"tensor width: {width} must be divisible by (3 * n_heads): {3 * self.n_heads}"
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=2)
        scale = 1 / np.sqrt(ch)
        weight = torch.einsum(
            "bghct,bghcs->bghts",
            (q * scale).view(bs, groups, self.n_heads, ch, length),
            (k * scale).view(bs, groups, self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bghts,bghcs->bghct", weight, v.reshape(bs, groups, self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


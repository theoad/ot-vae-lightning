from typing import Union, Sequence
from torch import Tensor
import torch.nn as nn
from ot_vae_lightning.utils.collage import Collage


class UnNormalize(nn.Module):
    """Scriptable output transform"""
    def __init__(
            self,
            mean: Union[Tensor, Sequence[float]],
            std: Union[Tensor, Sequence[float]],
            inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        mean = mean if isinstance(mean, Tensor) else Tensor(mean)
        mean = mean.view(1, -1, 1, 1) if mean.dim() == 1 else mean
        self.register_buffer("mean", mean)

        std = std if isinstance(std, Tensor) else Tensor(std)
        std = std.view(1, -1, 1, 1) if std.dim() == 1 else std
        self.register_buffer("std", std)

    def forward(self, x: Tensor) -> Tensor:
        if not self.inplace:
            x = x.clone()
        x.mul_(self.std).add_(self.mean)
        return x

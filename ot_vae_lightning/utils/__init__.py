from typing import Union, Sequence
from torch import Tensor
import torchvision.transforms as T
import torch.nn as nn
from ot_vae_lightning.utils.collage import Collage
from numpy import prod


class ToTensor(T.ToTensor):
    def __call__(self, pic):
        if isinstance(pic, Tensor): return pic
        return super().__call__(pic)


class UnNormalize(nn.Module):
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
        x.mul_(self.std.to(x.device)).add_(self.mean.to(x.device))
        return x


def squeeze_first_dims(batch):
    return batch.reshape(batch.shape[0] * batch.shape[1], *batch.shape[2:])


def unsqueeze_first_dims(batch, n):
    if n == 0 or n == 1 or batch is None:
        return batch
    return batch.reshape(n, batch.shape[0] // n, *batch.shape[1:])


def replicate_batch(batch, n):
    if n == 0 or n == 1 or batch is None:
        return batch
    return squeeze_first_dims(batch.unsqueeze(0).expand(n, *([-1] * len(batch.shape))))


def mean_replicated_batch(expanded_batch, n):
    if n == 0 or n == 1:
        return expanded_batch
    return unsqueeze_first_dims(expanded_batch, n).mean(0)


def std_replicated_batch(expanded_batch, n):
    if n == 0 or n == 1:
        return expanded_batch
    return unsqueeze_first_dims(expanded_batch, n).std(0)

import re
import inspect
from copy import copy

from typing import Union, Sequence, Callable, List
import contextlib
import numpy as np

import torch
from torch import Tensor
import torchvision.transforms as T
import torch.nn as nn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities import rank_zero_info
from ot_vae_lightning.utils.collage import *
from ot_vae_lightning.utils.elr import *
from ot_vae_lightning.utils.partial_checkpoint import *


class ToTensor(T.ToTensor):
    """
    TODO
    """
    def __call__(self, pic):
        if isinstance(pic, Tensor): return pic
        return super().__call__(pic)


class UnNormalize(nn.Module):
    """
    TODO
    """
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


class FilterKwargs(contextlib.AbstractContextManager):
    """
    Context manager that emulates a function which filters pre-defined
    key-worded arguments.

    Example:
        def func(arg1, arg2):
             ...

        # instead of this very verbose if-else statement:
        if hasarg(func, 'arg3'):
             res = func(x, y, z)
        else:
             res = func(x, y)

        # This class offers the following syntactic sugar:
        with FilterKwargs(func, 'z') as f:
            res = f(x, y, z=z)
    """
    def __init__(self, callee: Callable, arg_keys: Union[str, List[str]] = ()):
        self.callee = callee
        self.arg_keys = arg_keys if isinstance(arg_keys, list) else [arg_keys]

    def __call__(self, *args, **kwargs):
        filtered_kwargs = copy(kwargs)
        for key in self.arg_keys:
            if not hasarg(self.callee, key) and key in kwargs.keys():
                filtered_kwargs.pop(key)
        return self.callee(*args, **filtered_kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class debug(contextlib.ContextDecorator):
    """
    TODO
    """
    def __enter__(self):
        rank_zero_info('Entering')

    def __exit__(self, exc_type, exc, exc_tb):
        import IPython; IPython.embed(); exit(1)


def squeeze_first_dims(batch):
    """
    TODO
    :param batch:
    :return:
    """
    return batch.reshape(batch.shape[0] * batch.shape[1], *batch.shape[2:])


def unsqueeze_first_dims(batch, n):
    """
    TODO
    :param batch:
    :param n:
    :return:
    """
    if n == 0 or n == 1 or batch is None:
        return batch
    return batch.reshape(n, batch.shape[0] // n, *batch.shape[1:])


def replicate_tensor(t, n):
    """
    TODO
    :param t:
    :param n:
    :return:
    """
    return squeeze_first_dims(t.unsqueeze(0).expand(n, *([-1] * len(t.shape))))


def replicate_batch(batch, n):
    """
    TODO
    :param batch:
    :param n:
    :return:
    """
    if n == 0 or n == 1 or batch is None:
        return batch
    return apply_to_collection(batch, torch.Tensor, replicate_tensor, n)


def mean_replicated_batch(expanded_batch, n):
    """
    TODO
    :param expanded_batch:
    :param n:
    :return:
    """
    if n == 0 or n == 1:
        return expanded_batch
    return unsqueeze_first_dims(expanded_batch, n).mean(0)


def std_replicated_batch(expanded_batch, n):
    """
    TODO
    :param expanded_batch:
    :param n:
    :return:
    """
    if n == 0 or n == 1:
        return expanded_batch
    return unsqueeze_first_dims(expanded_batch, n).std(0)


def ema_inplace(moving_avg, new, decay):
    """
    TODO
    :param moving_avg:
    :param new:
    :param decay:
    :return:
    """
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    """
    TODO
    :param x:
    :param n_categories:
    :param eps:
    :return:
    """
    return (x + eps) / (x.sum() + n_categories * eps)


def hasarg(callee, arg_name: str):
    """
    TODO
    :param callee:
    :param arg_name:
    :return:
    """
    func = getattr(callee, 'forward') if isinstance(callee, nn.Module) else callee
    callee_params = inspect.signature(func).parameters.keys()
    return arg_name in callee_params


def permute_and_flatten(
        x: Tensor,
        permute_dims: Sequence[int],
        batch_first: bool = True,
        flatten_batch: bool = False
) -> Tensor:
    """
    TODO
    """
    remaining_dims = set(range(1, x.dim())).difference(set(permute_dims))
    if len(remaining_dims) == 0: return x.unsqueeze(0)

    if batch_first: x_rearranged = x.permute(0, *remaining_dims, *permute_dims)
    else:           x_rearranged = x.permute(*remaining_dims, 0, *permute_dims)

    x_rearranged = x_rearranged.flatten(
        int(batch_first and not flatten_batch),
        len(remaining_dims) - int(not batch_first and not flatten_batch))
    if flatten_batch: x_rearranged = x_rearranged.unsqueeze(0)
    return x_rearranged.contiguous()


def unflatten_and_unpermute(
        xr: Tensor,
        orig_shape: Sequence[int],
        permute_dims: Sequence[int],
        batch_first: bool = True,
        flatten_batch: bool = False
) -> Tensor:
    """
    TODO
    """
    remaining_dims = set(range(1, len(orig_shape))).difference(set(permute_dims))
    if len(remaining_dims) == 0: return xr.squeeze(0)

    if flatten_batch:
        xr = xr.squeeze(0)
        bs = orig_shape[0]
        n_elem_remaining = np.prod([orig_shape[d] for d in remaining_dims])
        xr = xr.unflatten(0, [bs, n_elem_remaining] if batch_first else [n_elem_remaining, bs])

    x = xr.unflatten(int(batch_first), [orig_shape[d] for d in remaining_dims])  # type: ignore[arg-type]

    permutation_map = list(range(len(orig_shape)))
    if not batch_first: permutation_map[0] = len(remaining_dims)

    for dim in range(1, len(orig_shape)):
        if dim in remaining_dims:
            permutation_map[dim] = list(remaining_dims).index(dim) + int(batch_first)
        else:
            permutation_map[dim] = len(remaining_dims) + 1 + permute_dims.index(dim)
    x = x.permute(*permutation_map)
    return x.contiguous()


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]


def camel2snake(name):
    return name[0].lower() + re.sub(r'(?!^)[A-Z]', lambda x: '_' + x.group(0).lower(), name[1:])


def removesuffix(string, suffix):
    if string.lower().endswith(suffix.lower()):
        return string[:-len(suffix)]
    return string

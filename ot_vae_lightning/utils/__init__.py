import re
import inspect
from copy import copy
from functools import partial
import warnings

from typing import Union, Sequence, Callable, List, Optional
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

from pytorch_lightning.utilities.distributed import sync_ddp_if_available, gather_all_tensors, distributed_available
from pytorch_lightning.utilities import rank_zero_warn

def gather_all_if_ddp_available(tensors: Tensor):
    if distributed_available():
        return gather_all_tensors(tensors)
    return [tensors]

DDPAllGather = Callable[[Tensor], List[Tensor]]
DDPAllReduce = Callable[[Tensor], Tensor]
DDPWarn = Callable[[str], None]
ddp_reduce_func_default: DDPAllReduce = partial(sync_ddp_if_available, reduce_op="sum")
ddp_gather_all_default: DDPAllGather = gather_all_if_ddp_available
ddp_warn_default: DDPWarn = rank_zero_warn


class DDPMixin(object):
    def __init__(
            self,
            ddp_reduce_func: Optional[DDPAllReduce] = ddp_reduce_func_default,
            ddp_gather_func: Optional[DDPAllGather] = gather_all_if_ddp_available,
            ddp_warn_func: Optional[DDPWarn] = ddp_warn_default
    ):
        self.reduce = ddp_reduce_func or (lambda x: x)
        self.gather = ddp_gather_func or (lambda x: [x])
        self.warn = ddp_warn_func or warnings.warn


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
    if decay is None:
        moving_avg.add_(new)
    else:
        moving_avg.mul_(decay).add_(new, alpha=(1 - decay))


def ema(moving_avg, new, decay):
    if decay is None: return moving_avg + new
    return moving_avg * decay + new * (1 - decay)


def laplace_smoothing(x, n_categories, eps=1e-5):
    """
    TODO
    :param x:
    :param n_categories:
    :param eps:
    :return:
    """
    if eps is None: return x
    return (x + eps) / (x.sum(-1, keepdim=True) + n_categories * eps) * x.sum(-1, keepdim=True)


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
    TL; DR
    >>> B, D1, B1, D2, B2, B3 = 10, 1, 2, 3, 4, 5
    >>> x = torch.randn(B, D1, B1, D2, B2, B3)
    >>> permute_and_flatten(x, (1, 3)).shape == torch.Size([B, B1 * B2 * B3, D1 * D2])
    >>> True
    >>> permute_and_flatten(x, (1, 3), batch_first=False).shape == torch.Size([B1 * B2 * B3, B, D1 * D2])
    >>> True
    >>> permute_and_flatten(x, (1, 3), flatten_batch=True).shape == torch.Size([B * B1 * B2 * B3, D1 * D2])
    """
    all_dims = set(range(1, x.dim()))
    if len(all_dims) == 0:
        raise ValueError("`input` is expected to have at least 2 dimensions")
    if len(permute_dims) == 0:
        raise ValueError("`permute_dims` is expected to contain at least one dimension")
    if not set(permute_dims).issubset(all_dims):
        raise ValueError("`permute_dims` is expected to be a subset of the `input` dimensions")

    remaining_dims = all_dims.difference(set(permute_dims))
    if len(remaining_dims) == 0: return x.flatten(int(not flatten_batch))

    if batch_first: x_rearranged = x.permute(0, *remaining_dims, *permute_dims).contiguous()
    else:           x_rearranged = x.permute(*remaining_dims, 0, *permute_dims).contiguous()

    x_rearranged = x_rearranged.flatten(
        int(batch_first and not flatten_batch),
        len(remaining_dims) - int(not batch_first and not flatten_batch))
    x_rearranged = x_rearranged.flatten(-len(permute_dims))
    return x_rearranged


def unflatten_and_unpermute(
        xr: Tensor,
        orig_shape: Sequence[int],
        permute_dims: Sequence[int],
        batch_first: bool = True,
        flatten_batch: bool = False
) -> Tensor:
    """
    TL; DR
    >>> B, D1, B1, D2, B2, B3 = 10, 1, 2, 3, 4, 5
    >>> x = torch.randn(B, D1, B1, D2, B2, B3)
    >>> xr = permute_and_flatten(x, permute_dims=(1, 3))
    >>> xr.shape == torch.Size([B, B1 * B2 * B3, D1 * D2])
    >>> True
    >>> xo = unflatten_and_unpermute(xr, x.shape, (1, 3))
    >>> bool((x == xo).all())
    >>> True
    """
    remaining_dims = set(range(1, len(orig_shape))).difference(set(permute_dims))
    if len(remaining_dims) == 0: return xr.view(*orig_shape)
    permute_shape = [orig_shape[d] for d in permute_dims]
    remaining_shape = [orig_shape[d] for d in remaining_dims]

    x = xr
    if flatten_batch:
        bs = orig_shape[0]
        n_elem_remaining = np.prod(remaining_shape)
        x = x.unflatten(0, [bs, n_elem_remaining] if batch_first else [n_elem_remaining, bs])

    x = x.unflatten(-1, permute_shape)
    x = x.unflatten(int(batch_first), remaining_shape)

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

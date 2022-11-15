import inspect
from copy import copy

from typing import Union, Sequence, Callable, List, Tuple
import contextlib

import torch
from torch import Tensor
import torchvision.transforms as T
import torch.nn as nn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities import rank_zero_info
from ot_vae_lightning.utils.collage import Collage


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
    def __enter__(self):
        rank_zero_info('Entering')

    def __exit__(self, exc_type, exc, exc_tb):
        import IPython; IPython.embed(); exit(1)


def squeeze_first_dims(batch):
    return batch.reshape(batch.shape[0] * batch.shape[1], *batch.shape[2:])


def unsqueeze_first_dims(batch, n):
    if n == 0 or n == 1 or batch is None:
        return batch
    return batch.reshape(n, batch.shape[0] // n, *batch.shape[1:])


def replicate_tensor(t, n):
    return squeeze_first_dims(t.unsqueeze(0).expand(n, *([-1] * len(t.shape))))


def replicate_batch(batch, n):
    if n == 0 or n == 1 or batch is None:
        return batch
    return apply_to_collection(batch, torch.Tensor, replicate_tensor, n)


def mean_replicated_batch(expanded_batch, n):
    if n == 0 or n == 1:
        return expanded_batch
    return unsqueeze_first_dims(expanded_batch, n).mean(0)


def std_replicated_batch(expanded_batch, n):
    if n == 0 or n == 1:
        return expanded_batch
    return unsqueeze_first_dims(expanded_batch, n).std(0)


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def hasarg(callee, arg_name: str):
    func = getattr(callee, 'forward') if isinstance(callee, nn.Module) else callee
    callee_params = inspect.signature(func).parameters.keys()
    return arg_name in callee_params


def permute_and_flatten(x: Tensor, permute_dims: Sequence[int], batch_first: bool = True) -> Tensor:
    remaining_dims = set(range(1, x.dim())).difference(set(permute_dims))
    if len(remaining_dims) == 0: return x.unsqueeze(0)

    if batch_first: x_rearranged = x.permute(0, *remaining_dims, *permute_dims)
    else:           x_rearranged = x.permute(*remaining_dims, 0, *permute_dims)

    x_rearranged = x_rearranged.flatten(int(batch_first), len(remaining_dims) - int(not batch_first))
    return x_rearranged.contiguous()


def unflatten_and_unpermute(
        xr: Tensor,
        orig_shape: Sequence[int],
        permute_dims: Sequence[int],
        batch_first: bool = True
) -> Tensor:
    remaining_dims = set(range(1, len(orig_shape))).difference(set(permute_dims))
    if len(remaining_dims) == 0: return xr.squeeze(0)

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

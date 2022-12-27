"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a data diffusion callback to gradually sharpen
the data distribution on which the model is trained

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Union, Any, Type, Dict, Sequence, Callable
import functools
from torchvision.transforms import Compose
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_warn

__all__ = ['NOOP', 'PgTransform', 'PgCompose', 'ProgressiveTransform']


class NOOP:
    def __call__(self, arg):
        return arg


class PgTransform:
    def __init__(self, transform_cls: Type, varying_kwargs: Dict[str, Sequence[Any]], **kwargs):
        self.transform_cls = transform_cls
        self.varying_kwargs = varying_kwargs
        self.kwargs = kwargs
        self.num_steps = max([len(seq) for seq in varying_kwargs.values()])

    def __getitem__(self, timestamp: int) -> Callable:
        if timestamp > self.num_steps:
            return NOOP()

        step_kwargs = {}
        for k, seq in self.varying_kwargs.items():
            step_kwargs[k] = seq[min(len(seq) - 1, timestamp)]

        transform = self.transform_cls(**step_kwargs, **self.kwargs)
        return transform


class PgCompose:
    def __init__(self, diffused_transforms: Sequence[PgTransform], compose_cls: Any = Compose):
        self.transforms = diffused_transforms
        self.compose_cls = compose_cls

    def __getitem__(self, timestamp: int) -> Callable:
        return self.compose_cls([t for t in self.transforms[timestamp]])


class ProgressiveTransform(Callback):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a data diffusion callback to gradually
    sharpen the data distribution on which the model is trained

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(
            self,
            transform: Union[PgTransform, PgCompose],
            schedule: Sequence[int],
    ):
        """
        :param transform: the progressive transforms to apply.
        :param schedule: list of epochs on which the transforms progress
        """
        super().__init__()
        self.transform = transform
        self.schedule = schedule

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch not in self.schedule:
            return

        found = False
        for func in dir(pl_module):
            method = getattr(pl_module, func)
            if callable(method) and hasattr(method, '__wrapped__') and hasattr(method.__wrapped__, 'transform'):
                method.__wrapped__.transform = self.transform[trainer.current_epoch]
                found = True

        if not found:
            rank_zero_warn("""
            `ProgressiveCallback` didn't find any method of the pl_module which should have its arguments transformed.
            Use @ProgressiveCallback.transform_arguments decorator in order to have a method affected by the callback.
            """)


def transform_args(getter_func: Callable = lambda x: x, setter_func: Callable = lambda orig, changed: orig):
    def decorator(method):
        method.transform = NOOP()

        @functools.wraps(method)
        def wrapper(self, *arg):
            to_transform = getter_func(*arg)
            transformed = method.transform(to_transform)
            new = setter_func(transformed, *arg)
            return method(self, new)
        return wrapper
    return decorator


transform_batch_tv = functools.partial(
    transform_args,
    getter_func=lambda b: b[0],
    setter_func=lambda x, b: (x, *b[1:])
)


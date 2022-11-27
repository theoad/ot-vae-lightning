"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a collage callback

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import os
import warnings
from typing import Any, Optional
import torch
import torchvision.utils
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


class Collage(Callback):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a collage callback to easily log images
    to tensorboard and wandb at the end of validation and test steps.

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(self, log_interval: int = 100, num_samples: int = 8) -> None:
        """
        :param log_interval: Number of steps between logging. Default: ``100``.
        :param num_samples: Number of images of displayed in the grid. Default: ``8``.
        """
        super().__init__()
        self.log_interval = log_interval
        self.num_samples = num_samples

    @staticmethod
    def log_method(method):
        """
        Decorates a method to mark it as a method which outputs a list of images to log
        """
        method.is_collage = True
        return method

    def log_images(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, mode: str = 'val'):
        found = False

        for func in dir(pl_module):
            method = getattr(pl_module, func)
            if not (callable(method) and hasattr(method, 'is_collage') and method.is_collage): continue
            found = True
            images = method(pl_module.batch_preprocess(batch))
            if len(images) == 0: continue
            collage = self.list_to_collage(images, self.num_samples)
            if trainer.logger is None:
                warnings.warn('No logger found. Logging locally.')
                if not os.path.exists('collages'): os.mkdir('collages')
                torchvision.utils.save_image(collage, f'collages/{str(trainer.global_step).zfill(4)}_{func}.png')
            elif isinstance(trainer.logger, WandbLogger):
                trainer.logger.log_image(   # type: ignore[arg-type]
                    f'{mode}/collage/{func}', [collage], step=trainer.global_step
                )
            elif isinstance(trainer.logger, TensorBoardLogger):
                trainer.logger.experiment.add_image(    # type: ignore[arg-type]
                    f'{mode}/collage/{func}', collage, global_step=trainer.global_step
                )
            else:
                raise NotImplementedError(f"Image logging for class {type(trainer.logger)} not supported")

        if not found:
            rank_zero_warn("""
            `Collage` didn't find any method of the pl_module which is marked a `collage_method` and should have its
             outputs logged. Use the @Collage.log_method decorator in order to have a method affected by the callback.
            """)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if batch_idx == 0 and trainer.is_global_zero:
            self.log_images(trainer, pl_module, batch, 'val')

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if batch_idx == 0 and trainer.is_global_zero:
            self.log_images(trainer, pl_module, batch, 'test')

    @staticmethod
    def list_to_collage(images, num_samples):
        if len(images) == 0:
            return
        elif len(images) == 1:
            collage_tensor = images[0].clamp(0, 1)
        else:
            collage_tensor = torch.cat(images, dim=-1).clamp(0, 1)  # concatenate on width dimension
        collage = make_grid(collage_tensor[:min(collage_tensor.size(0), num_samples)], nrow=1)
        return collage

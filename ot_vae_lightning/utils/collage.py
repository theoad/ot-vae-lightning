"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a collage callback

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Any, Optional
import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import Callback


class Collage(Callback):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a collage callback
    Adapted from `Lightning Bolts example
    <https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/callbacks/vision/sr_image_logger.py>`_

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

    def log_images(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, mode: str = 'val'):
        for method in pl_module.collage_methods():
            inputs = pl_module.batch_preprocess(batch)
            img_list = getattr(pl_module, method)(inputs)
            post_process = getattr(trainer.datamodule, f"{'train' if mode == 'train' else 'test'}_inverse_transform")
            if len(img_list) == 0:
                return
            collage_tensor = torch.cat(img_list, dim=-1).clamp(0, 1)  # concatenate on width dimension
            collage = make_grid(collage_tensor[:min(collage_tensor.size(0), self.num_samples)], nrow=1)
            trainer.logger.log_image(f'{mode}/collage', [collage], trainer.global_step, caption=method)

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
            self.log_images(trainer, pl_module, batch, 'test')

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
            self.log_images(trainer, pl_module, batch, 'val')

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        if trainer.global_step % trainer.log_every_n_steps == 0 and trainer.is_global_zero:
            pl_module.eval()
            with torch.no_grad():
                    self.log_images(trainer, pl_module, batch, 'train')
            pl_module.train()

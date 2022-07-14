"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an Abstract DataModule

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import os
from abc import ABC
import torch
from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class BaseDatamodule(pl.LightningDataModule, ABC):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract DataModule for MNIST

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad

    """
    def __init__(self,
                 train_batch_size: int = 32,
                 val_batch_size: int = 256,
                 test_batch_size: int = 256,
                 num_workers: int = 10,
                 ) -> None:
        """
        Lightning DataModule form MNIST dataset

        :param train_batch_size: Training batch size
        :param val_batch_size: Validation batch size
        :param test_batch_size: Testing batch size
        :param num_workers: Number of CPUs available
        """
        super().__init__()
        self.save_hyperparameters()
        self.transforms = None
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def _dataloader(self, mode):
        return DataLoader(
            getattr(self, f'{mode}_ds'),
            batch_size=getattr(self.hparams, f'{mode}_batch_size'),
            num_workers=self.hparams.num_workers,
            pin_memory=True,  # must pin memory for DDP
            shuffle=True if mode == 'train' else False
        )

    def train_dataloader(self):
        return self._dataloader('train')

    def val_dataloader(self):
        return self._dataloader('val')

    def test_dataloader(self):
        return self._dataloader('test')

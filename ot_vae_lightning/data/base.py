"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an Abstract DataModule

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from abc import ABC
from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import pytorch_lightning as pl


class BaseDatamodule(pl.LightningDataModule, ABC):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract DataModule

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad

    """
    def __init__(self,
                 train_transform: callable = T.ToTensor(),
                 test_transform: callable = T.ToTensor(),
                 train_val_split: float = 0.9,
                 seed: Optional[int] = None,
                 train_batch_size: int = 32,
                 val_batch_size: int = 256,
                 test_batch_size: int = 256,
                 num_workers: int = 10,
                 ) -> None:
        """
        Lightning DataModule

        :param train_transform: Transforms to apply on the train images
        :param test_transform: Transforms to apply on the val/test images
        :param train_val_split: Train-validation split coefficient
        :param seed: integer seed for re reproducibility
        :param train_batch_size: Training batch size
        :param val_batch_size: Validation batch size
        :param test_batch_size: Testing batch size
        :param num_workers: Number of CPUs available
        """
        super().__init__()
        self.save_hyperparameters()
        self.train_transform, self.test_transform = train_transform, test_transform
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def _dataloader(self, mode: str):
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

    @staticmethod
    def _dataset_split(dataset, split_prob=0.9, seed=None):
        if split_prob < 1:
            size = int(len(dataset) * split_prob)
            seed_generator = torch.Generator().manual_seed(seed) if seed is not None else None
            split = [size, len(dataset) - size]
            return random_split(dataset, split, seed_generator)

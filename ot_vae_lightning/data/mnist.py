"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of DataModule for MNIST

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import os
import torch
from typing import Optional
from ot_vae_lightning.data import BaseDatamodule
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
import torchvision.transforms as T
from torchvision.datasets import MNIST


class MNISTDatamodule(BaseDatamodule):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of DataModule for MNIST

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad

    """
    def __init__(self,
                 root: str = os.path.expanduser("~/.cache"),
                 no_pad: bool = False,
                 train_val_split: float = 0.8,
                 seed: Optional[int] = None,
                 train_batch_size: int = 32,
                 val_batch_size: int = 256,
                 test_batch_size: int = 256,
                 num_workers: int = 10,
                 ) -> None:
        """
        Lightning DataModule form MNIST dataset

        :param root: Path to folder where data will be stored
        :param no_pad: Set in order to keep image 28x28 (otherwise images will padded to 32x32)
        :param train_val_split: Train-validation split coefficient
        :param seed: integer seed for re reproducibility
        :param train_batch_size: Training batch size
        :param val_batch_size: Validation batch size
        :param test_batch_size: Testing batch size
        :param num_workers: Number of CPUs available
        """
        super().__init__()
        self.save_hyperparameters()
        self.transforms = None
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def prepare_data(self) -> None:
        MNIST(self.hparams.root, download=True, train=True)
        MNIST(self.hparams.root, download=True, train=False)

    def setup(self, stage: Optional[str] = None) -> None:
        transforms = T.ToTensor() if self.hparams.no_pad else T.Compose([T.Pad(2), T.ToTensor()])
        self.train_ds = MNIST(self.hparams.root, download=False, train=True, transform=transforms)
        self.test_ds = MNIST(self.hparams.root, download=False, train=False, transform=transforms)
        if self.hparams.train_val_split < 1:
            train_size = int(len(self.train_ds) * self.hparams.train_val_split)
            seed_generator = torch.Generator().manual_seed(self.hparams.seed) if self.hparams.seed is not None else None
            split = [train_size, len(self.train_ds) - train_size]
            self.train_ds, self.val_ds = random_split(self.train_ds, split, seed_generator)

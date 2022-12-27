"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of DataModule for torchvision datasets

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Optional, Union, Tuple
from enum import Enum

import torch
from ot_vae_lightning.data.base import BaseDatamodule, dataset_split
import torchvision.transforms as T
import torchvision.datasets as datasets
import inspect

__all__ = ['TorchvisionDatamodule']

TvDataset = Enum('TvDataset', datasets.__all__)


class TorchvisionDatamodule(BaseDatamodule):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of DataModule for torchvision datasets

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad

    """

    IMG_SIZE = None

    # noinspection PyUnusedLocal
    def __init__(self,
                 dataset_name: TvDataset,
                 root: Union[str, Tuple[str, str]],
                 download: bool = True,
                 num_workers: int = 10,
                 train_transform: callable = T.ToTensor(),
                 val_transform: callable = T.ToTensor(),
                 test_transform: callable = T.ToTensor(),
                 predict_transform: callable = T.ToTensor(),
                 inference_preprocess: callable = torch.nn.Identity(),
                 inference_postprocess: callable = torch.nn.Identity(),
                 test_val_split: float = 0.9,
                 seed: Optional[int] = None,
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 test_batch_size: int = 32,
                 predict_batch_size: int = 32,
                 ) -> None:
        """
        Lightning DataModule form MNIST dataset

        :param dataset_name: Name of the dataset. Default: 'MNIST'
        :param root: Path to folder where data will be stored
        :param download: If ``True``, the dataset is downloaded in `self.prepare_data`
        :param num_workers: Number of CPUs available
        :param train_transform: Transforms to apply on the `train` images
        :param val_transform: Transforms to apply on the `validation` images
        :param test_transform: Transforms to apply on the `test` images
        :param predict_transform: Transforms to apply on the `predict` images
        :param inference_preprocess: Transform to apply on raw inference data (that did not go through train_transform)
        :param inference_postprocess: used to reverse `preprocess` in inference, visualization (e.g. denormalize images)
        :param test_val_split: Test-validation split coefficient
        :param seed: integer seed for re reproducibility
        :param train_batch_size: Training batch size
        :param val_batch_size: Validation batch size
        :param test_batch_size: Testing batch size
        :param predict_batch_size: Predict batch size
        """
        super().__init__(num_workers, train_transform, val_transform, test_transform, predict_transform,
                         inference_preprocess, inference_postprocess, seed, train_batch_size,
                         val_batch_size, test_batch_size, predict_batch_size)
        self.dataset_cls = getattr(datasets, dataset_name)

        # Some datasets have arg `split` and others `train`
        # e.g. ImageNet(.., split='train') and MNIST(.., train=True)
        signature = inspect.signature(self.dataset_cls.__init__)
        if 'split' in signature.parameters.keys():
            self._train_kwarg = dict(split='train')
            self._test_kwarg = dict(split='val')
        elif 'train' in signature.parameters.keys():
            self._train_kwarg = dict(train=True)
            self._test_kwarg = dict(train=False)
        else:
            self._train_kwarg, self._test_kwarg = {}, {}

    def prepare_data(self) -> None:
        if self.hparams.download:
            self.dataset_cls(self.hparams.root, download=True, **self._train_kwarg)
            self.dataset_cls(self.hparams.root, download=True, **self._test_kwarg)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = self.dataset_cls(
            self.hparams.root[0] if isinstance(self.hparams.root, tuple) else self.hparams.root,
            transform=self.train_transform,
            **self._train_kwarg
        )

        self.val_dataset = self.test_dataset = self.dataset_cls(
            self.hparams.root[1] if isinstance(self.hparams.root, tuple) else self.hparams.root,
            transform=self.val_transform,
            **self._test_kwarg
        )

        self.val_dataset, self.test_dataset = dataset_split(
            datasets=[self.val_dataset, self.test_dataset],
            split=self.hparams.test_val_split,
            seed=self.hparams.seed
        )

        self.predict_dataset = None

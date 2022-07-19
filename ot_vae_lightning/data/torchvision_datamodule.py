"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of DataModule for torchvision datasets

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Optional, Union, Tuple
from ot_vae_lightning.data import BaseDatamodule
import torchvision.transforms as T
import torchvision.datasets as datasets
import inspect


class TorchvisionDatamodule(BaseDatamodule):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of DataModule for torchvision datasets

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad

    """
    def __init__(self,
                 dataset_name: str,
                 root: Union[str, Tuple[str, str]],
                 train_transform: callable = T.ToTensor(),
                 test_transform: callable = T.ToTensor(),
                 train_val_split: float = 0.9,
                 seed: Optional[int] = None,
                 train_batch_size: int = 32,
                 val_batch_size: int = 256,
                 test_batch_size: int = 256,
                 num_workers: int = 10,
                 download: bool = True,
                 ) -> None:
        """
        Lightning DataModule form MNIST dataset

        :param dataset_name: Name of the dataset. Default: 'MNIST'
        :param root: Path to folder where data will be stored
        :param train_transform: Transforms to apply on the train images
        :param test_transform: Transforms to apply on the val/test images
        :param train_val_split: Train-validation split coefficient
        :param seed: integer seed for re reproducibility
        :param train_batch_size: Training batch size
        :param val_batch_size: Validation batch size
        :param test_batch_size: Testing batch size
        :param num_workers: Number of CPUs available
        :param download: If ``True``, the dataset is downloaded in `self.prepare_data`
        """
        super().__init__(train_transform, test_transform, train_val_split, seed, train_batch_size, val_batch_size,
                         test_batch_size, num_workers)
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
        self.train_ds = self.dataset_cls(
            self.hparams.root[0] if isinstance(self.hparams.root, tuple) else self.hparams.root,
            transform=self.train_transform,
            **self._train_kwarg
        )

        self.train_ds, self.val_ds = self._dataset_split(self.train_ds, self.hparams.train_val_split, self.hparams.seed)
        if hasattr(self.val_ds, 'transforms'): self.val_ds.transforms = self.test_transform

        self.test_ds = self.dataset_cls(
            self.hparams.root[0] if isinstance(self.hparams.root, tuple) else self.hparams.root,
            transform=self.test_transform,
            **self._test_kwarg
        )

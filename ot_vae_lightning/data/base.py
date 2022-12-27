"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an Abstract DataModule

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from abc import ABC
from typing import Optional, Union, Sequence, Callable
import torch
from torch import randperm
from torch.utils.data import DataLoader, Dataset, Subset
from itertools import accumulate
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule

__all__ = ['BaseDatamodule', 'dataset_split']


class BaseDatamodule(LightningDataModule, ABC):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract DataModule

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """

    # noinspection PyUnusedLocal
    def __init__(self,
                 num_workers: int = 10,
                 train_transform: Callable = T.ToTensor(),
                 val_transform: Callable = T.ToTensor(),
                 test_transform: Callable = T.ToTensor(),
                 predict_transform: Callable = T.ToTensor(),
                 inference_preprocess: Callable = lambda x: x,
                 inference_postprocess: Callable = lambda x: x,
                 seed: Optional[int] = None,
                 train_batch_size: int = 32,
                 val_batch_size: int = 256,
                 test_batch_size: int = 256,
                 predict_batch_size: int = 256,
                 **dataloader_kwargs
                 ) -> None:
        """
        Lightning DataModule

        :param train_transform: Transforms to apply on the `train` images
        :param val_transform: Transforms to apply on the `validation` images
        :param test_transform: Transforms to apply on the `test` images
        :param predict_transform: Transforms to apply on the `predict` images
        :param inference_preprocess: Transform to apply on raw inference data (that did not go through train_transform)
        :param inference_postprocess: used to reverse `preprocess` in inference, visualization (e.g. denormalize images)
        :param seed: integer seed for re reproducibility
        :param train_batch_size: Training batch size
        :param val_batch_size: Validation batch size
        :param test_batch_size: Testing batch size
        :param predict_batch_size: Predict batch size
        :param num_workers: Number of CPUs available
        """
        super().__init__()
        self.save_hyperparameters(ignore=[
            'train_transform',
            'val_transform',
            'test_transform',
            'predict_transform',
            'inference_preprocess',
            'inference_postprocess'
        ])
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.predict_transform = predict_transform
        self.inference_preprocess = inference_preprocess
        self.inference_postprocess = inference_postprocess
        self.train_dataset, self.val_dataset, self.test_dataset, self.predict_dataset = None, None, None, None
        self.dataloader_kwargs = dataloader_kwargs

    def _dataloader(self, mode: str):
        kwargs = {
            'num_workers': self.hparams.num_workers,
            'pin_memory': True,  # must pin memory for DDP
            'shuffle': True if mode == 'train' else False,
            **self.dataloader_kwargs  # will override the params above
        }

        return DataLoader(
            getattr(self, f'{mode}_dataset'),
            batch_size=getattr(self.hparams, f'{mode}_batch_size'),
            **kwargs
        )

    def train_dataloader(self):
        return self._dataloader('train')

    def val_dataloader(self):
        return self._dataloader('val')

    def test_dataloader(self):
        return self._dataloader('test')

    def predict_dataloader(self):
        return self._dataloader('predict')


def dataset_split(
        datasets: Sequence[Dataset],
        split: Union[Sequence[int], float] = 0.9,
        seed: Optional[int] = None
):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results.
    Adapted from torch.utils.data.random_split to allow for different transform in each split

    Args:
        datasets (sequence of Dataset): Datasets to be split
        split (sequence, float): sequence of length or proportion of splits (for a 2-fold split)
        seed (int): seed used for the random permutation reproducibility.
    """
    length = len(datasets[0])   # type: ignore[arg-type]
    for d in datasets:
        assert len(d) == length, f"The datasets are expected to all have the same size. Found {length} and {len(d)}"  # type: ignore[arg-type]

    if isinstance(split, float):
        if split > 1 or split < 0:
            raise ValueError(f"The split probability must verify 0 <= split_prob <= 1. Given: {split}")

        size = int(length * split)
        split = [size, length - size]

    # Cannot verify that dataset is Sized
    if sum(split) != length:  # type: ignore[arg-type]
        raise ValueError(f"Sum of input lengths does not equal the length of the input datasets! "
                         f"Given length={length} and split={split}")

    seed_generator = torch.Generator().manual_seed(seed) if seed is not None else None
    indices = randperm(sum(split), generator=seed_generator).tolist()
    return [
        Subset(d, indices[offset - length: offset]) for d, offset, length in zip(datasets, accumulate(split), split)
    ]

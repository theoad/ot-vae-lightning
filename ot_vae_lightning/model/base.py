"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract module

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import os
from typing import Optional, Dict
from collections import OrderedDict
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection


class PartialCheckpoint:
    """
    --> Why add a checkpoint loading methods on top of lightning's --ckpt_path ?
    --> lightning's --ckpt_path loads weights for all attributes in a strict manner as well as
        hparams, optimizer states, metric states and so on.
        This custom checkpoint loading method loads only weights for specific attributes.
    """
    def __init__(self, checkpoint_path: str, attr_name: str = None, replace_str: str = ""):
        self.attr_name = attr_name
        self.checkpoint_path = checkpoint_path
        self.replace_str = replace_str
        # assert os.path.exists(checkpoint_path), f'Error: Path {checkpoint_path} not found.'

    def state_dict(self):
        checkpoint = torch.load(self.checkpoint_path)
        if self.attr_name is None or all([self.attr_name not in k for k in checkpoint['state_dict'].keys()]):
            return checkpoint['state_dict']

        state_dict = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            if self.attr_name == '.'.join(key.split('.')[:self.attr_name.count('.')+1]):
                state_dict[key.replace(f"{self.attr_name}.", self.replace_str)] = checkpoint['state_dict'][key]

        return state_dict


class BaseModule(pl.LightningModule, ABC):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract module

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(
            self,
            metrics: Optional[MetricCollection] = None,
            out_transforms: Optional[callable] = None,
            checkpoints: Optional[Dict[str, PartialCheckpoint]] = None,
    ):
        """
        ----------------------------------------------------------------------------------------------------------------

        DO Initialize here::

            - [Required]: Assign loss function to self.loss
            - [Optional]: Fields that contain nn.Parameters (e.g. nn.Modules)

        DON'T Initialize here::

            - [Required]: Metrics - instead pass as argument to super().__init__()
            - [Optional]: hyper-parameter attributes (e.g. self.lr=lr) - instead use self.save_hyperparameters
            - [Optional]: Datasets - implement datasets and dataloader related logic in a separated pl.DataModule.

        ----------------------------------------------------------------------------------------------------------------

        :param metrics: torchmetrics.MetricCollection duplicated for train, validation and test
        :param out_transforms: transforms used to reverse the test transform for visualization (e.g. denormalize images)
        :param checkpoints: dictionary of attribute <-> Partial checkpoint (e.g. {'encoder': PartialCheckpoint}
        """
        super().__init__()
        # as nn.Module, metrics are automatically saved on checkpointing, so we don't save them as hparams
        self.save_hyperparameters()

        self.loss = ...    # assign the loss function. (can be a list for alternate updates like GANs)

        self.train_metrics = metrics.clone(prefix='train/metrics/') if metrics is not None else None
        self.val_metrics = metrics.clone(prefix='val/metrics/') if metrics is not None else None
        self.test_metrics = metrics.clone(prefix='test/metrics/') if metrics is not None else None

        # post processing like un-normalize. Must be scriptable (no PIL transforms nor lambda functions)
        self.out_transforms = torch.nn.Identity() if out_transforms is None else out_transforms
        self.checkpoints = checkpoints

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def batch_preprocess(self, batch):
        """
        Pre-process batch before feeding to self.loss and computing metrics.

        :param batch: Output of self.train_ds.__getitem__()
        :return: Dictionary (or any key-valued container) with at least the keys `samples` and `targets`
        """
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.loss[optimizer_idx] if hasattr(self.loss, '__getitem__') else self.loss
        loss, logs, pbatch = loss(self.batch_preprocess(batch), batch_idx)
        metric_result = self.train_metrics(pbatch['preds'], pbatch['targets'])
        self.log_dict({**logs, **metric_result}, sync_dist=False, rank_zero_only=True, prog_bar=True, logger=True)
        return loss

    def setup(self, stage=None):
        # Checkpoint loading. Let you load partial attributes.
        if self.checkpoints is not None:
            for attr, partial_ckpt in self.checkpoints.items():
                state_dict = partial_ckpt.state_dict()
                getattr(self, attr).load_state_dict(state_dict, strict=True)
                print(f'[info]: self.{attr} loaded successfully')

    def _prepare_metrics(self, mode):
        for metric in getattr(self, f'{mode}_metrics').values():
            if hasattr(metric, 'prepare_metric'):
                metric.prepare_metric(self)
                self.print(f'{mode}_{metric} preparation done')

    def _update_metrics(self, batch, batch_idx, mode):
        pbatch = self.batch_preprocess(batch)
        if 'preds' not in pbatch.keys(): pbatch['preds'] = self(pbatch['samples'])
        getattr(self, f'{mode}_metrics').update(pbatch['preds'], pbatch['targets'])
        return pbatch

    def _compute_and_log_metric(self, mode):
        res = getattr(self, f'{mode}_metrics').compute()
        getattr(self, f'{mode}_metrics').reset()
        self.log_dict(res, sync_dist=True, prog_bar=True, logger=True)
        return res

    def on_fit_start(self) -> None:
        return self._prepare_metrics('val')

    def on_test_start(self) -> None:
        return self._prepare_metrics('test')

    def validation_step(self, batch, batch_idx):
        return self._update_metrics(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._update_metrics(batch, batch_idx, 'test')

    def validation_epoch_end(self, outputs):
        return self._compute_and_log_metric('val')

    def test_epoch_end(self, outputs):
        return self._compute_and_log_metric('test')

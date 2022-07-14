"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract module

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import os
from typing import Union, Optional
from collections import OrderedDict
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl
from torchmetrics import Metric, MetricCollection


class BaseModule(pl.LightningModule, ABC):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract module

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(self, metrics: Union[Metric, MetricCollection], checkpoint: Optional[str] = None):
        """
        DO Initialize here::

            - Fields that don't contain nets.Parameters
            - Assign loss function to self.loss

        DON'T Initialize here::

            - nets.Modules - instead override self._init_modules()
            - Datasets - instead override self._init_datasets()
            - Metrics - instead override self._init_metrics()
        """
        super().__init__()
        # as nn.Module, metrics are automatically saved on checkpointing, so we don't save them as hparams
        self.save_hyperparameters(ignore=['metrics'])

        self.loss = ...  # assign the loss function. (can be a list for alternate updates like GANs)

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

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
        self._log_dict({**logs, **self.train_metrics(pbatch['preds'], pbatch['targets'])})
        return loss

    def _update_metrics(self, batch, batch_idx, mode):
        pbatch = self.batch_preprocess(batch)
        if 'preds' not in pbatch.keys(): pbatch['preds'] = self(pbatch['samples'])
        getattr(self, f'{mode}_metrics').update(pbatch['preds'], pbatch['targets'])
        return pbatch

    def _log_dict(self, logs):
        slash_logs = {}
        for key in logs.keys(): slash_logs[key.replace("_", "/")] = logs[key]
        # TODO: consider synd_dist=False and rank_zero_only=True
        self.log_dict(slash_logs, sync_dist=True, prog_bar=True)
        return slash_logs

    def _log_metric(self, mode):
        res = getattr(self, f'{mode}_metrics').compute()
        getattr(self, f'{mode}_metrics').reset()
        return self._log_dict(res)

    def _load_attr_state_dict(self, attr):
        if getattr(self, attr) is None or not isinstance(getattr(self, attr), torch.nn.Module):
            return
        assert os.path.exists(self.hparams.checkpoint), f'Error: Path {self.hparams.checkpoint} not found.'
        checkpoint = torch.load(self.hparams.checkpoint)
        state_dict = OrderedDict()
        found = False
        for key, val in checkpoint['state_dict'].items():
            if attr == key.split('.')[0]:
                found = True
                state_dict['.'.join(key.split('.')[1:])] = val
        if found:
            getattr(self, attr).load_state_dict(state_dict, strict=False)

    def setup(self, stage=None):
        # Checkpoint loading. Let you load partial attributes.
        if self.hparams.checkpoint is not None:
            for attr, _ in self.named_children():
                self._load_attr_state_dict(attr)

    def _prepare_metrics(self, mode):
        for metric in getattr(self, f'{mode}_metrics').values():
            if hasattr(metric, 'prepare_metric'):
                metric.prepare_metric(self)

    def on_fit_start(self) -> None:
        return self._prepare_metrics('val')

    def on_test_start(self) -> None:
        return self._prepare_metrics('test')

    def validation_step(self, batch, batch_idx):
        return self._update_metrics(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._update_metrics(batch, batch_idx, 'test')

    def validation_epoch_end(self, outputs):
        return self._log_metric('val')

    def test_epoch_end(self, outputs):
        return self._log_metric('test')

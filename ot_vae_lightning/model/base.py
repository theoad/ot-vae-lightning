"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract module

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import os
import functools
from typing import Optional, Dict, Any
from collections import OrderedDict
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import get_model_size_mb
from torchmetrics import MetricCollection


class PartialCheckpoint:
    """
    --> Why add a checkpoint loading methods on top of lightning's --ckpt_path ?
    --> lightning's --ckpt_path loads weights for all attributes in a strict manner as well as
        hparams, optimizer states, metric states and so on.
        This custom checkpoint loading method loads only weights for specific attributes.
    """
    def __init__(
            self,
            checkpoint_path: str,
            attr_name: str = None,
            replace_str: str = "",
            strict: bool = True,
            freeze: bool = False
    ):
        self.attr_name = attr_name
        self.checkpoint_path = checkpoint_path
        self.replace_str = replace_str
        self.strict = strict
        self.freeze = freeze
        assert os.path.exists(checkpoint_path), f'Error: Path {checkpoint_path} not found.'

    @property
    def state_dict(self):
        checkpoint = torch.load(self.checkpoint_path)
        if self.attr_name is None or all([self.attr_name not in k for k in checkpoint['state_dict'].keys()]):
            return checkpoint['state_dict']

        state_dict = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            if self.attr_name == '.'.join(key.split('.')[:self.attr_name.count('.')+1]):
                state_dict[key.replace(f"{self.attr_name}.", self.replace_str)] = checkpoint['state_dict'][key]

        return state_dict


def support_preprocess(method):
    @functools.wraps(method)
    def preprocess(self, samples, *args, no_preprocess_override=False, **kwargs):
        if self.inference and not no_preprocess_override:
            samples = self.inference_preprocess(samples)
        return method(self, samples, *args, **kwargs)
    return preprocess


def support_postprocess(method):
    @functools.wraps(method)
    def postprocess(self, *args, no_postprocess_override=False, **kwargs):
        out = method(self, *args, **kwargs)
        if self.inference and not no_postprocess_override:
            out = self.inference_postprocess(out)
        return out
    return postprocess


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

        :param metrics: Metrics (torchmetrics.MetricCollection) to use for train, validation and test
        :param checkpoints: Dictionary of: [attribute <-> Partial checkpoint]
        (e.g. {'encoder': PartialCheckpoint('my_autoencoder_checkpoint', replace_str='autoencoder.encoder')})
        """
        super().__init__()
        self.checkpoints = checkpoints

        self.loss = ...    # assign the loss function. (can be a list for alternate updates like GANs)

        self.train_metrics = metrics.clone(prefix='train/metrics/') if metrics is not None else None
        self.val_metrics = metrics.clone(prefix='val/metrics/') if metrics is not None else None
        self.test_metrics = metrics.clone(prefix='test/metrics/') if metrics is not None else None

        self.inference_preprocess = None  # to be populated by checkpoint loading
        self.inference_postprocess = None  # to be populated by checkpoint loading
        self._inference_flag = False

    @abstractmethod
    def batch_preprocess(self, batch: Any) -> Dict[str, Any]:
        """
        Pre-process batch before feeding to self.loss and computing metrics.

        :param batch: Output of self.train_ds.__getitem__()
        :return: Dictionary (or any key-valued container) with at least the keys `samples` and `targets`
        """
        pass

    @support_preprocess
    @support_postprocess
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Implement as a regular forward
        """
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.loss[optimizer_idx] if hasattr(self.loss, '__getitem__') else self.loss
        loss, logs, pbatch = loss(self.batch_preprocess(batch), batch_idx)
        metric_result = self.train_metrics(pbatch['preds'], pbatch['targets'])
        full_log = {**logs, **metric_result, 'loss': loss}
        self.log_dict(full_log, sync_dist=False, rank_zero_only=True, prog_bar=True, logger=True)
        return {**full_log, **pbatch}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self._update_metrics(batch, batch_idx, 'val')

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self._update_metrics(batch, batch_idx, 'test')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch = self.batch_preprocess(batch)
        out = self.forward(batch['samples'])  # type: ignore[arg-type]
        batch['preds'] = out
        return out

    def validation_epoch_end(self, outputs):
        return self._compute_and_log_metric('val')

    def test_epoch_end(self, outputs):
        return self._compute_and_log_metric('test')

    def on_train_epoch_start(self) -> None:
        self.inference = False

    def on_validation_epoch_start(self) -> None:
        self.inference = True

    def on_test_epoch_start(self) -> None:
        self.inference = True

    def on_predict_epoch_start(self) -> None:
        self.inference = True

    def on_fit_start(self) -> None:
        self._set_inference_transforms()
        return self._prepare_metrics('train')

    def on_validation_start(self) -> None:
        self._set_inference_transforms()
        return self._prepare_metrics('val')

    def on_test_start(self) -> None:
        self._set_inference_transforms()
        return self._prepare_metrics('test')

    @abstractmethod
    def configure_optimizers(self):
        pass

    def setup(self, stage=None):
        # Checkpoint loading. Let you load partial attributes.
        if self.checkpoints is not None:
            for attr, partial_ckpt in self.checkpoints.items():
                getattr(self, attr).load_state_dict(partial_ckpt.state_dict, strict=partial_ckpt.strict)
                print(f'[info]: self.{attr}[{int(get_model_size_mb(getattr(self, attr)))}Mb] loaded successfully.')

    def _prepare_metrics(self, mode):
        for metric in getattr(self, f'{mode}_metrics').values():
            if hasattr(metric, 'prepare_metric'):
                metric.prepare_metric(self)
                self.print(f'{mode}_{metric} preparation done')

    def _update_metrics(self, batch, batch_idx, mode):
        pbatch = self.batch_preprocess(batch)
        pbatch['preds'] = self(pbatch['samples'])
        getattr(self, f'{mode}_metrics').update(pbatch['preds'], pbatch['targets'])
        return pbatch

    def _compute_and_log_metric(self, mode):
        res = getattr(self, f'{mode}_metrics').compute()
        getattr(self, f'{mode}_metrics').reset()
        self.log_dict(res, sync_dist=True, prog_bar=True, logger=True)
        return res

    def _set_inference_transforms(self) -> None:
        if self.trainer is None or self.trainer.datamodule is None:  # type: ignore[arg-type]
            return
        dm = self.trainer.datamodule  # type: ignore[arg-type]
        if self.inference_preprocess is None and hasattr(dm, 'inference_preprocess'):
            self.inference_preprocess = dm.inference_preprocess
        if self.inference_postprocess is None and hasattr(dm, 'inference_postprocess'):
            self.inference_postprocess = dm.inference_postprocess

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.inference_preprocess is not None:
            checkpoint['inference_preprocess'] = self.inference_preprocess
        if self.inference_postprocess is not None:
            checkpoint['inference_postprocess'] = self.inference_postprocess

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if 'inference_preprocess' in checkpoint.keys():
            self.inference_preprocess = checkpoint['inference_preprocess']
        if 'inference_postprocess' in checkpoint.keys():
            self.inference_postprocess = checkpoint['inference_postprocess']

    @property
    def inference(self):
        return self._inference_flag

    @inference.setter
    def inference(self, boolean: bool):
        if boolean:
            assert self.inference_preprocess is not None and self.inference_postprocess is not None,\
                'Tried to set model in inference mode but ' \
                'self.inference_preprocess or self.inference_postprocess in None'
        self._inference_flag = boolean

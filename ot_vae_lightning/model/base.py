"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract module

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Optional, Dict, Any, Callable, Literal
from abc import ABC, abstractmethod
import importlib.util
import functools

import torch
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint, RichProgressBar, RichModelSummary, TQDMProgressBar, ModelSummary
)
from pytorch_lightning.cli import LightningCLI
from torchmetrics import MetricCollection
from torch_ema import ExponentialMovingAverage
from ot_vae_lightning.utils.partial_checkpoint import PartialCheckpoint
from ot_vae_lightning.utils import Collage

__all__ = ['VisionModule', 'VisionCLI']


class VisionModule(pl.LightningModule, ABC):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an abstract module

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(
            self,
            metrics: MetricCollection,
            monitor: str = "accuracy",
            mode: Literal['min', 'max'] = "min",
            checkpoints: Optional[Dict[str, PartialCheckpoint]] = None,
            metric_on_train: bool = False,
            inference_preprocess: Optional[Callable] = None,
            inference_postprocess: Optional[Callable] = None,
            ema_decay: Optional[float] = None,
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
        :param monitor: The name of the metric to monitor for early stopping and checkpoint saving (should be a key in `metrics`)
        :param mode: Indicates whether the monitored metric should be minized or maximized
        :param checkpoints: Dictionary of: [attribute <-> Partial checkpoint]
                            (e.g. {'encoder': PartialCheckpoint('my_encoder_checkpoint', replace_str='encoder')})
        :param metric_on_train: If ``True``, will duplicate the validation/test metrics and use the synchronized
                                per-step metric computation feature of torchmetrics
                                (TL;DR: call `forward` on self.train_metrics).
        :param inference_preprocess: Transform to apply on raw inference data (that did not go through train_transform)
        :param inference_postprocess: used to reverse `preprocess` in inference, visualization (e.g. denormalize images)
        :param ema_decay: If set, will use as decay parameter for exponential moving averaging the model weights
        """
        super().__init__()
        self.checkpoints = checkpoints

        self.loss = ...    # assign the loss function. (can be a list for alternate updates like GANs)

        self.val_metrics = metrics.clone(prefix='val/metrics/') if metrics is not None else None
        self.test_metrics = metrics.clone(prefix='test/metrics/') if metrics is not None else None
        self.train_metrics = metrics.clone(prefix='train/metrics/') if metrics is not None and metric_on_train else None
        self.monitor = self.val_metrics.prefix + monitor
        self.mode = mode

        self.inference_preprocess = inference_preprocess
        self.inference_postprocess = inference_postprocess
        self._inference_flag = False
        self.ema_decay = ema_decay
        self._ema = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Same as :meth:`torch.nn.Module.forward()`.

        For more details, see pl.LightningModule documentation.
        """

    @abstractmethod
    def batch_preprocess(self, batch: Any) -> Dict[str, Any]:
        """
        Pre-process batch before feeding to self.loss and computing metrics.

        :param batch: Output of self.train_ds.__getitem__()
        :return: Dictionary (or any key-valued container) with at least the keys `samples` and `target`.
        """

    def optim_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.loss[optimizer_idx] if hasattr(self.loss, '__getitem__') else self.loss
        loss, logs, pbatch = loss(self.batch_preprocess(batch), batch_idx)
        if self.train_metrics is not None:
            metric_result = self.train_metrics(pbatch['preds'], pbatch['target'])
            logs = {**logs, **metric_result}
        self.log_dict(logs, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=False)
        return {'loss': loss, **logs, **pbatch}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self._update_metrics(batch, batch_idx, 'val')

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self._update_metrics(batch, batch_idx, 'test')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch = self.batch_preprocess(batch)
        kwargs = batch['kwargs'] if 'kwargs' in batch else {}
        out = self(batch['samples'], **kwargs)
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
        if self._ema is not None: self._ema.store(); self._ema.copy_to()

    def on_test_epoch_start(self) -> None:
        self.inference = True
        if self._ema is not None: self._ema.store(); self._ema.copy_to()

    def on_predict_epoch_start(self) -> None:
        self.inference = True
        if self._ema is not None: self._ema.store(); self._ema.copy_to()

    def on_validation_epoch_end(self) -> None:
        if self._ema is not None: self._ema.restore()

    def on_test_epoch_end(self) -> None:
        if self._ema is not None: self._ema.restore()

    def on_predict_epoch_end(self, results) -> None:
        if self._ema is not None: self._ema.restore()

    def on_fit_start(self) -> None:
        self._set_inference_transforms()
        if self.ema_decay is not None:
            self._ema = ExponentialMovingAverage(self.optim_parameters(), decay=self.ema_decay)
        return self._prepare_metrics('train')

    def on_validation_start(self) -> None:
        self._set_inference_transforms()
        return self._prepare_metrics('val')

    def on_test_start(self) -> None:
        self._set_inference_transforms()
        return self._prepare_metrics('test')

    def on_before_zero_grad(self, optimizer) -> None:
        if self._ema is not None:
            self._ema.update(self.optim_parameters())

    def setup(self, stage=None):
        if self.checkpoints is not None:
            for attr, partial_ckpt in self.checkpoints.items():
                partial_ckpt.load_attribute(self, attr)

    @torch.no_grad()
    def _prepare_metrics(self, mode):
        if getattr(self, f'{mode}_metrics') is None: return
        for metric_name, metric in getattr(self, f'{mode}_metrics').items():
            if hasattr(metric, 'prepare_metric'):
                metric.prepare_metric(self)
                self.print(f'{mode}_{metric_name} preparation done')

    @torch.no_grad()
    def _update_metrics(self, batch, batch_idx, mode):
        if getattr(self, f'{mode}_metrics') is None: return
        pbatch = self.batch_preprocess(batch)
        kwargs = pbatch['kwargs'] if 'kwargs' in pbatch else {}
        pbatch['preds'] = self(pbatch['samples'], **kwargs)
        if hasattr(self, 'sample'): pbatch['generated'] = self.sample(pbatch['samples'].size(0), **kwargs)
        getattr(self, f'{mode}_metrics').update(**pbatch)
        return pbatch

    def _compute_and_log_metric(self, mode):
        if getattr(self, f'{mode}_metrics') is None: return
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
            if self.inference_preprocess is None or self.inference_postprocess is None:
                self._set_inference_transforms()
            assert self.inference_preprocess is not None,\
                'Tried to set model in inference mode but self.inference_preprocess is None'
            assert self.inference_postprocess is not None,\
                'Tried to set model in inference mode but self.inference_postprocess is None'
        self._inference_flag = boolean

    @staticmethod
    def preprocess(method):
        """
        Class method decorator to automatically pre-process inputs using
        inference transforms. The transforms are loaded from the datamodule
        that served in train. They are saved together with the checkpoint and
        automatically loaded together with the model.
        Useful for methods that process data like `forward`, `encode`, `predict`, ...
        """
        @functools.wraps(method)
        def wrapper(self, samples, *args, no_preprocess_override=False, **kwargs):
            if self.inference and not no_preprocess_override:
                samples = self.inference_preprocess(samples)
            return method(self, samples, *args, **kwargs)
        return wrapper

    @staticmethod
    def postprocess(method):
        """
        Class method decorator to automatically post-process outputs using
        inference transforms. The transforms are loaded from the datamodule
        that served in train. They are saved together with the checkpoint and
        automatically loaded together with the model.
        Useful for methods that output data like `forward`, `decode`, `sample`, ...
        """
        @functools.wraps(method)
        def wrapper(self, *args, no_postprocess_override=False, **kwargs):
            out = method(self, *args, **kwargs)
            if self.inference and not no_postprocess_override:
                if isinstance(out, list):
                    out = [self.inference_postprocess(o) for o in out]
                else:
                    out = self.inference_postprocess(out)
            return out
        return wrapper


class VisionCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.inference_preprocess", "model.inference_preprocess", apply_on="instantiate")
        parser.link_arguments("data.inference_postprocess", "model.inference_postprocess", apply_on="instantiate")

        parser.set_defaults({
            "trainer.accelerator": "gpu",
            "trainer.devices": 1,
            "trainer.strategy": None,
            "trainer.benchmark": True,
            "trainer.precision": 32,
            "trainer.max_epochs": 100,
            "trainer.gradient_clip_val": None,
        })

        parser.add_lightning_class_args(Collage, "collage")
        parser.set_defaults({
            "collage.log_interval": 100,
            "collage.num_samples": 8
        })

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({
            "early_stopping.monitor": None,
            "early_stopping.mode": None,
            "early_stopping.min_delta": 0.1,
            "early_stopping.patience": 5,
            "early_stopping.verbose": True,
            "early_stopping.log_rank_zero_only": True
        })

        parser.add_lightning_class_args(ModelCheckpoint, "checkpointing")
        parser.set_defaults({
            "checkpointing.save_top_k": 10,
            "checkpointing.monitor": None,
            "checkpointing.mode": None,
            "checkpointing.filename": None,
        })

        parser.link_arguments("model.monitor", "early_stopping.monitor", apply_on="instantiate")
        parser.link_arguments("model.monitor", "checkpointing.monitor",  apply_on="instantiate")
        parser.link_arguments("model.mode", "early_stopping.mode", apply_on="instantiate")
        parser.link_arguments("model.mode", "checkpointing.mode",  apply_on="instantiate")
        parser.link_arguments(
            "trainer.logger.init_args.name", "checkpointing.filename",
            compute_fn=lambda name: name + "-{epoch:02d}-{val_loss:.2f}",
        )

        # rich_installed = importlib.util.find_spec("rich") is not None
        # parser.add_lightning_class_args(RichProgressBar if rich_installed else TQDMProgressBar, "pbar")
        # parser.add_lightning_class_args(RichModelSummary if rich_installed else ModelSummary, "summary")

    def instantiate_trainer(self, **kwargs: Any) -> pl.Trainer:
        trainer = super().instantiate_trainer(**kwargs)
        # if isinstance(trainer.logger, WandbLogger):
        #     wandb.save(self.save_config_filename)
        return trainer

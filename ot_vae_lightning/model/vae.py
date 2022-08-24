"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a VAE

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import warnings
from typing import Tuple, Optional, Dict, List, Union, Callable
import functools
import numpy as np

import wandb
import torch
from torch import Tensor
from torchvision.transforms.transforms import GaussianBlur
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics import MetricCollection
from ot_vae_lightning.prior import Prior
from ot_vae_lightning.model.base import VisionModule, VisionCLI
from ot_vae_lightning.utils import Collage
from ot_vae_lightning.ot import LatentTransport
import ot_vae_lightning.utils as utils
from ot_vae_lightning.data import ProgressiveTransform, PgTransform
import ot_vae_lightning.data.progressive_callback as progressive
from ot_vae_lightning.utils.partial_checkpoint import PartialCheckpoint
import ot_vae_lightning.data    # noqa: F401


class VAE(VisionModule):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a VAE

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """

    Batch = Dict[str, Union[Tensor, Dict]]
    Conditional = utils.FilterKwargs
    Conditional.__init__ = functools.partialmethod(Conditional.__init__, arg_keys='labels')

    # noinspection PyUnusedLocal
    def __init__(self,
                 prior: Prior,
                 autoencoder: Optional[nn.Module] = None,
                 encoder: Optional[nn.Module] = None,
                 decoder: Optional[nn.Module] = None,
                 metrics: Optional[MetricCollection] = None,
                 checkpoints: Optional[Dict[str, PartialCheckpoint]] = None,
                 inference_preprocess: Optional[Callable] = None,
                 inference_postprocess: Optional[Callable] = None,
                 learning_rate: float = 1e-3,
                 lr_sched_metric: Optional[str] = None,
                 conditional: bool = False,
                 expansion: int = 1,
    ) -> None:
        """
        Variational Auto Encoder with custom Prior

        ------------------------------------------------------------------------------------

         .. code-block:: python

             model = VAE(
                metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
                encoder=CNN(1, 128, 32, 2, None, 4, 2, True, False),
                decoder=CNN(64, 1, 2, 32, None, 4, 2, False, True),
                prior=GaussianPrior()
             )

        ------------------------------------------------------------------------------------

        :param prior: prior class (derived from abstract class Prior)
        :param autoencoder: A nn.Module with methods `self.encode` and `self.decode`.
                            Can be left unfilled if `encoder and `decoder` parameter are filled.
        :param encoder: A nn.Module with method `self.encode`. Can be left unfilled if `autoencoder` is filled
        :param decoder: A nn.Module with method `self.decode`. Can be left unfilled if `autoencoder` is filled
        :param learning_rate: The model learning rate. Default `1e-3`
        :param checkpoints: See father class (ot_vae_lightning.model.base.BaseModule) docstring
        """
        super().__init__(metrics, checkpoints, True, inference_preprocess, inference_postprocess)
        if autoencoder is None and (encoder is None or decoder is None):
            raise ValueError("At least one of `autoencoder` or (`encoder`, `decoder`) parameters must be set")
        if autoencoder is not None and (encoder is not None or decoder is not None):
            raise ValueError("Setting both `autoencoder` and `encoder` or `decoder` is ambiguous")

        self.save_hyperparameters()

        self.loss = self.elbo
        self.prior = prior
        self._warn_conditional_call('prior.forward'); self._warn_conditional_call('prior.sample')

        if autoencoder is not None:
            assert (
                    isinstance(autoencoder, nn.Module) and
                    hasattr(autoencoder, 'encode') and hasattr(autoencoder, 'decode')
            ), 'Parameter `autoencoder` should be a nn.Module and implement the methods `encode` and `decode`'
            self.autoencoder = autoencoder
            self._encode_func = autoencoder.encode; self._warn_conditional_call('autoencoder.encode')
            self._decode_func = autoencoder.decode; self._warn_conditional_call('autoencoder.decode')
            # Don't set self.encoder and self.decoder in order for checkpoints loading to not be ambiguous
        else:
            assert encoder is not None
            assert decoder is not None
            self.encoder = encoder
            self.decoder = decoder
            self._encode_func = encoder; self._warn_conditional_call('encoder.forward')
            self._decode_func = decoder; self._warn_conditional_call('decoder.forward')
            # Don't set self.autoencoder in order for checkpoints loading to not be ambiguous

        self._expand = functools.partial(utils.replicate_batch, n=expansion)
        self._reduce_mean = functools.partial(utils.mean_replicated_batch, n=expansion)
        self._reduce_std = functools.partial(utils.std_replicated_batch, n=expansion)

    @progressive.transform_batch_tv()
    def batch_preprocess(self, batch) -> Batch:
        samples, labels = batch
        kwargs = {'labels': labels if self.hparams.conditional else None}
        return {
            'samples': samples,
            'targets': samples,
            'kwargs': kwargs
        }

    @VisionModule.postprocess
    @VisionModule.preprocess
    def forward(self, samples: Tensor, labels: Optional[Tensor] = None, expand: bool = False) -> Tensor:
        latents = self.encode(samples, labels, expand=expand, no_preprocess_override=True)
        reconstructions = self.decode(latents, labels, expand_y=expand, no_postprocess_override=True)
        return reconstructions

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.lr_sched_metric is None: return opt

        name = self.val_metrics.prefix + self.hparams.lr_sched_metric
        mode = 'max' if self.val_metrics[self.hparams.lr_sched_metric].higher_is_better else 'min'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=mode, factor=0.5, patience=5, verbose=True, threshold=1e-1, min_lr=1e-8
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": name}}

    # noinspection PyUnusedLocal
    def elbo(self, batch: Batch, batch_idx: int) -> Tuple[Tensor, Dict[str, float], Batch]:
        samples, labels, targets = batch['samples'], batch['kwargs']['labels'], batch['targets']
        batch_size = samples.size(0)

        latents, prior_loss = self.encode(samples, labels, expand=True, return_prior_loss=True)
        reconstructions = self.decode(latents, labels, expand_y=True)
        reconstructions_mean = self._reduce_mean(reconstructions)

        prior_loss = prior_loss.mean()
        recon_loss = (
            F.mse_loss(reconstructions_mean, targets) if self.prior.empirical_kl else
            F.mse_loss(reconstructions_mean, targets, reduction="none").sum(dim=(1, 2, 3)).mean()
        )
        loss = recon_loss + prior_loss
        logs = {
            'train/loss/total': loss.item(),
            'train/loss/recon': recon_loss.item(),
            'train/loss/prior': prior_loss.item()
        }
        batch['preds'] = reconstructions[:batch_size]
        batch['latents'] = latents[:batch_size]
        batch['preds_mean'] = reconstructions_mean
        return loss, logs, batch

    @property
    def latent_size(self):
        if hasattr(self, 'autoencoder'):
            enc_out = self.autoencoder.latent_size
        else:
            assert hasattr(self, 'encoder')
            enc_out = self.encoder.out_size
        return self.prior.out_size(enc_out)

    @VisionModule.preprocess
    def encode(
            self,
            samples: Tensor,
            labels: Optional[Tensor] = None,
            return_prior_loss: bool = False,
            expand: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        with self.Conditional(self._encode_func) as encode, self.Conditional(self.prior) as prior:
            encodings = encode(samples, labels=labels)
            if expand: encodings, labels = self._expand((encodings, labels))
            latents, prior_loss = prior(encodings, step=self.global_step, labels=labels)

        if return_prior_loss:
            assert not self.inference, 'The prior loss cannot be returned when the model is in inference mode'
            return latents, prior_loss

        return latents

    @VisionModule.postprocess
    def decode(self, latents: Tensor, labels: Optional[Tensor] = None, expand_y: bool = False) -> Tensor:
        if expand_y:
            labels = self._expand(labels)

        with VAE.Conditional(self._decode_func) as decode:
            return decode(latents, labels=labels)

    @VisionModule.postprocess
    def sample(self, batch_size: int, labels: Optional[Tensor] = None) -> Tensor:
        with VAE.Conditional(self.prior.sample) as sample:
            latents = sample((batch_size, *self.latent_size), device=self.device, labels=labels)

        return self.decode(latents, labels, no_postprocess_override=True)

    @Collage.log_method
    def reconstruction(self, batch: Batch) -> List[Tensor]:
        samples, labels, t = batch['samples'], batch['kwargs']['labels'], batch['targets']
        batch_size = samples.size(0)
        reconstructions = self(samples, labels, expand=True)
        reconstructions_mean = self._reduce_mean(reconstructions)
        reconstructions_std = self._reduce_std(reconstructions)
        realizations = [reconstructions[batch_size * i:batch_size * (i + 1)] for i in range(self.hparams.expansion)]
        return [t, reconstructions_mean, *realizations, reconstructions_std]

    @Collage.log_method
    def generation(self, batch: Batch) -> List[Tensor]:
        samples, labels = batch['samples'], batch['kwargs']['labels']
        return [self.sample(samples.size(0), labels) for _ in range(4)]

    def _warn_conditional_call(self, method) -> None:
        module = self
        for attr in method.rsplit('.'):
            module = getattr(module, attr)
        if self.hparams.conditional and not utils.hasarg(module, 'labels'):
            warnings.warn(f"`conditional` is specified but `{method}` doesn't accept `labels` parameter")


class VAECli(VisionCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)


if __name__ == '__main__':
    cli = VAECli(
        VAE,
        save_config_filename='cli_config.yaml',
        save_config_overwrite=True,
        run=False
    )

    if isinstance(cli.trainer.logger, WandbLogger):
        wandb.save(cli.save_config_filename)

    transport_kwargs = dict(
        size=cli.model.latent_size,
        transformations=GaussianBlur(5, sigma=(1.5, 1.5)),
        transport_type="gaussian",
        transformed_latents_from_train=True,
        make_pd=True,
        verbose=True,
        pg_star=0,
    )

    ProgressiveGaussianBlur = PgTransform(
        GaussianBlur, {'sigma': list(zip(np.linspace(10, 0, 50), np.linspace(10, 0, 50)))[:-1]}, kernel_size=5
    )

    callbacks = [
        Collage(),
        ProgressiveTransform(ProgressiveGaussianBlur, list(range(50))),
        LatentTransport(
            transport_dims=(1,),
            diag=False,
            stochastic=False,
            logging_prefix="mat_per_needle",
            **transport_kwargs
        ),
        LatentTransport(
            transport_dims=(1,),
            diag=True,
            stochastic=False,
            logging_prefix="mat_per_needle_diag",
            **transport_kwargs
        ),
    ]

    cli.trainer.callbacks += callbacks
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)

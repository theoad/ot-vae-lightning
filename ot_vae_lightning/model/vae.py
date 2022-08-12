"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a VAE

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Tuple, Optional, Dict, List, Union
from functools import partial

import wandb
import torch
from torch import Tensor
from torchvision.transforms.transforms import GaussianBlur
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection
from ot_vae_lightning.data import MNIST32, MNIST, CIFAR10, FFHQ128
from ot_vae_lightning.prior import Prior
from ot_vae_lightning.model.base import BaseModule, PartialCheckpoint, inference_preprocess, inference_postprocess
from ot_vae_lightning.utils import Collage
from ot_vae_lightning.ot import LatentTransport
import ot_vae_lightning.utils as utils


class VAE(BaseModule):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a VAE

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """

    Batch = Dict[str, Tensor]

    def __init__(self,
                 prior: Prior,
                 metrics: Optional[MetricCollection] = None,
                 checkpoints: Optional[Dict[str, PartialCheckpoint]] = None,
                 autoencoder: Optional[nn.Module] = None,
                 encoder: Optional[nn.Module] = None,
                 decoder: Optional[nn.Module] = None,
                 learning_rate: float = 1e-3,
                 plateau_metric_monitor: str = None,
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
        :param autoencoder: A nn.Module with methods `self.encode` and `self.decode`. Can be left unfilled if `encoder
                            and `decoder` parameter are filled.
        :param encoder: A nn.Module with method `self.encode`. Can be left unfilled if `autoencoder` is filled
        :param decoder: A nn.Module with method `self.decode`. Can be left unfilled if `autoencoder` is filled
        :param learning_rate: The model learning rate. Default `1e-3`
        :param checkpoints: See father class (ot_vae_lightning.model.base.BaseModule) docstring
        """
        super().__init__(metrics, checkpoints, metric_on_train=True)
        self.save_hyperparameters()

        if autoencoder is None and (encoder is None or decoder is None):
            raise ValueError("At least one of `autoencoder` or (`encoder`, `decoder`) parameters must be set")
        if autoencoder is not None and (encoder is not None or decoder is not None):
            raise ValueError("Setting both `autoencoder` and `encoder` or `decoder` is ambiguous")
        if autoencoder is not None:
            assert (
                    isinstance(autoencoder, nn.Module) and
                    hasattr(autoencoder, 'encode') and
                    hasattr(autoencoder, 'decode')
            ), f'Parameter `autoencoder` should be a nn.Module and implement the methods `encode` and `decode`'
            # Don't set self.encoder and self.decoder in order for partial checkpoints loading to not be ambiguous
            self.autoencoder = autoencoder
        else:
            assert encoder is not None
            assert decoder is not None
            # Don't set self.autoencoder in order for partial checkpoints loading to not be ambiguous
            self.encoder = encoder
            self.decoder = decoder

        self.prior = prior

        self.loss = self.elbo

        self._expand = partial(utils.replicate_batch, n=self.hparams.expansion)
        self._reduce_mean = partial(utils.mean_replicated_batch, n=self.hparams.expansion)
        self._reduce_std = partial(utils.std_replicated_batch, n=self.hparams.expansion)

    def batch_preprocess(self, batch) -> Batch:
        x, y = batch
        return {'samples': x, 'targets': x}

    @inference_postprocess
    @inference_preprocess
    def forward(self, x: Tensor, expand: bool = False) -> Tensor:
        z = self.encode(x, no_preprocess_override=True, expand=expand)
        x_hat = self.decode(z, no_postprocess_override=True)
        return x_hat

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.plateau_metric_monitor is None:
            return opt
        mode = 'max' if self.val_metrics[self.hparams.plateau_metric_monitor].higher_is_better else 'min'
        name = self.val_metrics.prefix + self.hparams.plateau_metric_monitor
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=mode, factor=0.5, patience=20, verbose=True, threshold=0.2
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": name}}

    def elbo(self, batch: Batch, batch_idx: int) -> Tuple[Tensor, Dict[str, float], Batch]:
        samples, targets = batch['samples'], batch['targets']
        batch_size = samples.size(0)

        z, prior_loss = self.encode(samples, return_prior_loss=True, expand=True)
        x_hat = self.decode(z)
        x_hat_mean = self._reduce_mean(x_hat)

        prior_loss = prior_loss.mean()
        if self.prior.empirical_kl:
            recon_loss = F.mse_loss(x_hat_mean, targets)
        else:
            recon_loss = F.mse_loss(x_hat_mean, targets, reduction="none").sum(dim=(1, 2, 3)).mean()

        loss = recon_loss + prior_loss
        logs = {
            'train/loss/total': loss.item(),
            'train/loss/recon': recon_loss.item(),
            'train/loss/prior': prior_loss.item()
        }
        batch['preds'], batch['latents'], batch['preds_mean'] = x_hat[:batch_size], z, x_hat_mean
        return loss, logs, batch

    @property
    def latent_size(self):
        if hasattr(self, 'autoencoder'):
            enc_out = self.autoencoder.latent_size
        else:
            assert hasattr(self, 'encoder')
            enc_out = self.encoder.out_size
        return self.prior.out_size(enc_out)

    @inference_preprocess
    def encode(self, x: Tensor, return_prior_loss: bool = False, expand: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if hasattr(self, 'autoencoder'):
            encodings = self.autoencoder.encode(x)
        else:
            assert hasattr(self, 'encoder')
            encodings = self.encoder(x)

        if expand:
            encodings = self._expand(encodings)

        z, prior_loss = self.prior(encodings, step=self.global_step)

        if return_prior_loss:
            assert not self.inference, 'The prior loss cannot be returned when the model is in inference mode'
            return z, prior_loss
        else:
            return z

    @inference_postprocess
    def decode(self, z: Tensor) -> Tensor:
        if hasattr(self, 'autoencoder'):
            return self.autoencoder.decode(z)
        else:
            assert hasattr(self, 'decoder')
            return self.decoder(z)

    @staticmethod
    def collage_methods() -> List[str]:
        return ['reconstruction', 'generation']

    @torch.no_grad()
    def reconstruction(self, batch: Batch) -> List[Tensor]:
        x, t = batch['samples'], batch['targets']
        x_hat = self(x, expand=True)
        x_hat_mean, x_hat_std = self._reduce_mean(x_hat), self._reduce_std(x_hat)
        realizations = [x_hat[x.size(0) * i:x.size(0) * (i + 1)] for i in range(self.hparams.expansion)]
        return [t, x_hat_mean] + realizations + [x_hat_std]

    @torch.no_grad()
    def generation(self, batch: Batch) -> List[Tensor]:
        batch_size = batch['samples'].shape[0]
        return [self.samples(batch_size) for _ in range(4)]

    @inference_postprocess
    def samples(self, batch_size: int) -> Tensor:
        z = self.prior.sample((batch_size, *self.latent_size), self.device)
        return self.decode(z, no_postprocess_override=True)


if __name__ == '__main__':
    cli = LightningCLI(
        VAE, MNIST,
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

    callbacks = [
        Collage(),
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

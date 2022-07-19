"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a VAE

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Tuple, Union, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.utilities.cli import LightningCLI
from torchmetrics import Metric, MetricCollection
from ot_vae_lightning.data import MNISTDatamodule
from ot_vae_lightning.prior import Prior
from ot_vae_lightning.model.base import BaseModule
from ot_vae_lightning.utils import Collage


class VAE(BaseModule):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a VAE

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(self,
                 metrics: Union[Metric, MetricCollection],
                 prior: Prior,
                 autoencoder: Optional[nn.Module] = None,
                 encoder: Optional[nn.Module] = None,
                 decoder: Optional[nn.Module] = None,
                 learning_rate: float = 1e-3,
                 checkpoint: Optional[str] = None,
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
        """
        super().__init__(metrics, checkpoint)
        self.save_hyperparameters(ignore=['metrics', 'prior', 'encoder', 'decoder', 'autoencoder'])
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
            assert encoder is not None and decoder is not None
            # Don't set self.encoder and self.decoder in order for partial checkpoints loading to not be ambiguous
            self.encoder = encoder
            self.decoder = decoder

        self.prior = prior
        self.loss = self.elbo

    def batch_preprocess(self, batch):
        x, y = batch
        return {'samples': x, 'targets': x}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if hasattr(self, 'autoencoder'):
            encodings = self.autoencoder.encode(x)
        else:
            assert hasattr(self, 'encoder')
            encodings = self.encoder(x)
        z, prior_loss = self.prior(encodings)
        return z, prior_loss

    def decode(self, z: Tensor) -> Tensor:
        if hasattr(self, 'autoencoder'):
            return self.autoencoder.decode(z)
        else:
            assert hasattr(self, 'decoder')
            return self.decoder(z)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, loss = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, loss

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[0]

    def elbo(self, batch, batch_idx):
        x = batch['targets']
        x_hat, prior_loss = self._forward(batch['samples'])
        batch['preds'] = x_hat

        prior_loss = prior_loss.mean()
        recon_loss = F.mse_loss(x_hat, x, reduction="none").sum(dim=(1, 2, 3)).mean()

        loss = recon_loss + prior_loss
        logs = {
            'train_loss_total': loss.item(),
            'train_loss_recon': recon_loss.item(),
            'train_loss_prior': prior_loss.item()
        }
        return loss, logs, batch

    @staticmethod
    def collage_methods():
        return ['reconstruction', 'generation']

    def reconstruction(self, batch):
        x = batch['targets']
        x_hat = self(batch['samples'])
        return [x, x_hat]

    def generation(self, batch):
        z, _ = self.encode(batch['samples'])
        z = self.prior.sample(z.shape, self.device)
        samples = self.decode(z)
        return [samples]


if __name__ == '__main__':
    callbacks = [Collage(), RichProgressBar()]
    cli = LightningCLI(VAE, MNISTDatamodule,
                       trainer_defaults=dict(default_root_dir='.', callbacks=callbacks),
                       run=False, save_config_overwrite=True)
    cli.trainer.logger.watch(cli.model, log="all")
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)

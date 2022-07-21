"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a VAE

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Tuple, Optional, Dict, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.utilities.cli import LightningCLI
from torchmetrics import MetricCollection
from ot_vae_lightning.data import MNISTDatamodule
from ot_vae_lightning.prior import Prior
from ot_vae_lightning.model.base import BaseModule, PartialCheckpoint
from ot_vae_lightning.utils import Collage


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
                 autoencoder: Optional[nn.Module] = None,
                 encoder: Optional[nn.Module] = None,
                 decoder: Optional[nn.Module] = None,
                 learning_rate: float = 1e-3,
                 out_transforms: Optional[nn.Module] = None,
                 checkpoints: Optional[Dict[str, PartialCheckpoint]] = None,
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
        :param out_transforms: Transforms used to reverse the test transform for visualization (e.g. denormalize images)
        :param checkpoints: See father class (ot_vae_lightning.model.base.BaseModule) docstring
        """
        super().__init__(metrics, out_transforms, checkpoints)
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
            assert encoder is not None and decoder is not None
            # Don't set self.encoder and self.decoder in order for partial checkpoints loading to not be ambiguous
            self.encoder = encoder
            self.decoder = decoder

        self.prior = prior
        self.loss = self.loss_func

    def batch_preprocess(self, batch) -> Batch:
        x, y = batch
        return {'samples': x, 'targets': x}

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[0]

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, loss = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def loss_func(self, batch: Batch, batch_idx: int) -> Tuple[Tensor, Dict[str, float], Batch]:
        x = batch['targets']
        x_hat, prior_loss = self._forward(batch['samples'])
        batch['preds'] = x_hat

        prior_loss = prior_loss.mean()
        recon_loss = F.mse_loss(x_hat, x, reduction="none").sum(dim=(1, 2, 3)).mean()

        loss = recon_loss + prior_loss
        logs = {
            'train/loss/total': loss.item(),
            'train/loss/recon': recon_loss.item(),
            'train/loss/prior': prior_loss.item()
        }
        return loss, logs, batch

    @property
    def latent_size(self):
        if hasattr(self, 'autoencoder'):
            enc_out = self.autoencoder.latent_size
        else:
            assert hasattr(self, 'encoder')
            enc_out = self.encoder.out_size
        return torch.Size((enc_out[0]//2, *enc_out[1:]))  # re-parametrization trick

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

    @staticmethod
    def collage_methods() -> List[str]:
        return ['reconstruction', 'generation']

    @ torch.no_grad()
    def reconstruction(self, batch: Batch) -> List[Tensor]:
        x = self.out_transforms(batch['targets'])
        x_hat = self.out_transforms(self(batch['samples']))
        return [x, x_hat]

    @ torch.no_grad()
    def generation(self, batch: Batch) -> List[Tensor]:
        batch_size = batch['samples'].shape[0]
        return [self.out_transforms(self.samples(batch_size)) for _ in range(4)]

    def samples(self, batch_size: int) -> Tensor:
        z = self.prior.sample((batch_size, *self.latent_size), self.device)
        return self.decode(z)


if __name__ == '__main__':
    callbacks = [Collage()]#, RichProgressBar()]
    cli = LightningCLI(VAE, MNISTDatamodule,
                       trainer_defaults=dict(default_root_dir='.', callbacks=callbacks),
                       run=False, save_config_overwrite=True)
    cli.trainer.logger.watch(cli.model, log="all")
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)

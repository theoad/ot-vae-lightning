"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a VAE

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Tuple, Union
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import LightningCLI
from torchmetrics import Metric, MetricCollection
from ot_vae_lightning.data import MNISTDatamodule
from ot_vae_lightning.prior import Prior
from ot_vae_lightning.model.base import BaseModule


class VAE(BaseModule):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a VAE

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(self,
                 metrics: Union[Metric, MetricCollection],
                 encoder: nn.Module,
                 decoder: nn.Module,
                 prior: Prior) -> None:
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

        :param encoder: VAE encoder architecture
        :param decoder: VAE decoder architecture
        :param prior: prior class (derived from abstract class Prior)
        """
        super().__init__(metrics)
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.loss = self.elbo

    def batch_preprocess(self, batch):
        x, y = batch
        return {'samples': x, 'targets': x}

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        encodings = self.encoder(x)
        z, prior_loss = self.prior(encodings)
        return z, prior_loss

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z, loss = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, loss

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[0]

    def elbo(self, batch, batch_idx):
        x = batch['targets']
        x_hat, prior_loss = self._forward(x)
        batch['preds'] = x_hat

        prior_loss = prior_loss.mean()
        recon_loss = F.mse_loss(x_hat, x)

        loss = recon_loss + prior_loss
        logs = {
            'train_loss_total': loss.item(),
            'train_loss_recon': recon_loss.item(),
            'train_loss_prior': prior_loss.item()
        }
        return loss, logs, batch


if __name__ == '__main__':
    LightningCLI(VAE, MNISTDatamodule)

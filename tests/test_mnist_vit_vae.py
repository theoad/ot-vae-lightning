"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a CI fot ViT VAE

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import torch
from pytorch_lightning import Trainer, seed_everything
from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from ot_vae_lightning.model import VAE
from ot_vae_lightning.prior import GaussianPrior
from ot_vae_lightning.data import MNIST
from ot_vae_lightning.networks import ViT

_PSNR_PERFORMANCE = 18
_MAX_EPOCH = 2


def test_vae_vit_training(prog_bar=False, gpus=None):
    seed_everything(42)

    trainer = Trainer(max_epochs=_MAX_EPOCH, enable_progress_bar=prog_bar, gpus=gpus)
    datamodule = MNIST(train_batch_size=50)

    encoder = ViT(28, 7, 64, 4, 4, 128, 1, 32, 0.1, 0., 2, None, 2, True, False)
    decoder = ViT(28, 7, 64, 2, 2, 128, 1, 32, 0.1, 0., None, 1, None, False, True)

    model = VAE(  # LightningModule
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=encoder,
        decoder=decoder,
        prior=GaussianPrior(loss_coeff=0.1, reparam_dim=1),
    )

    assert model.latent_size == torch.Size((1, 64))

    # Train
    trainer.fit(model, datamodule)
    trainer.save_checkpoint("vanilla_vae_vit.ckpt")

    # Test
    model.freeze()
    results = trainer.test(model, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE


if __name__ == "__main__":
    pbar, gpu = True, 1
    test_vae_vit_training(pbar, gpu)

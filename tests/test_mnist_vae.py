"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a CI

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from pytorch_lightning import Trainer, seed_everything
from ot_vae_lightning.model import VAE
from ot_vae_lightning.prior import GaussianPrior
from ot_vae_lightning.data import MNISTDatamodule
from ot_vae_lightning.networks import CNN, AutoEncoder
from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio


def test_vanilla_vae():
    seed_everything(42)

    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=CNN(1, 256, 32, 1, None, 8, 2, True, False, "batchnorm", "relu"),
        decoder=CNN(128, 1, 1, 32, None, 8, 2, False, True, "batchnorm", "relu"),
        prior=GaussianPrior(loss_coeff=0.1)
    )

    trainer = Trainer(limit_train_batches=250, limit_val_batches=40, max_epochs=5)
    datamodule = MNISTDatamodule(train_batch_size=250)
    trainer.fit(model, datamodule)
    results = trainer.test(model, datamodule)
    assert results[0]['test/psnr'] > 17

    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        autoencoder=AutoEncoder(1, 128, True, 32, 1, None, 8, 2, True, "batchnorm", "relu"),
        prior=GaussianPrior(loss_coeff=0.1)
    )

    trainer = Trainer(limit_train_batches=250, limit_val_batches=40, max_epochs=5)
    trainer.fit(model, datamodule)
    results = trainer.test(model, datamodule)
    assert results[0]['test/psnr'] > 17

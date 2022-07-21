"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a CI

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from pytorch_lightning import Trainer, seed_everything
from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from ot_vae_lightning.model import VAE, PartialCheckpoint
from ot_vae_lightning.prior import GaussianPrior
from ot_vae_lightning.data import MNISTDatamodule
from ot_vae_lightning.networks import CNN, AutoEncoder
from ot_vae_lightning.utils import UnNormalize

_PSNR_PERFORMANCE = 15
_MAX_EPOCH = 10


def test_vanilla_vae_encoder_decoder():
    seed_everything(42)

    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=CNN(1, 256, 32, 1, None, 8, 2, True, False, "batchnorm", "relu"),
        decoder=CNN(128, 1, 1, 32, None, 8, 2, False, True, "batchnorm", "relu"),
        prior=GaussianPrior(loss_coeff=0.1),
        out_transforms=UnNormalize((0.1307,), (0.3081,))
    )

    trainer = Trainer(limit_train_batches=250, limit_val_batches=40, max_epochs=_MAX_EPOCH, enable_progress_bar=False)
    datamodule = MNISTDatamodule(train_batch_size=250)
    trainer.fit(model, datamodule)
    trainer.save_checkpoint("vanilla_vae_encoder_decoder.ckpt")

    results = trainer.test(model, datamodule)
    print(results)

    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE

    vae = VAE.load_from_checkpoint("vanilla_vae_encoder_decoder.ckpt")
    vae.eval()

    results = trainer.test(vae, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE


def test_vanilla_vae_autoencoder():
    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        autoencoder=AutoEncoder(1, 128, True, 32, 1, None, 8, 2, True, "batchnorm", "relu"),
        prior=GaussianPrior(loss_coeff=0.1),
        out_transforms=UnNormalize((0.1307,), (0.3081,))
    )

    # Training
    trainer = Trainer(limit_train_batches=250, limit_val_batches=40, max_epochs=_MAX_EPOCH, enable_progress_bar=True)
    datamodule = MNISTDatamodule(train_batch_size=250)
    trainer.fit(model, datamodule)
    trainer.save_checkpoint("vanilla_vae_autoencoder.ckpt")

    results = trainer.test(model, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE

    # Checkpoint loading
    vae = VAE.load_from_checkpoint("vanilla_vae_autoencoder.ckpt")
    vae.eval()

    results = trainer.test(vae, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE

    # Partial checkpoint loading
    trainer.save_checkpoint("vanilla_vae_autoencoder.ckpt", weights_only=True)

    checkpoints = dict(
        encoder=PartialCheckpoint("vanilla_vae_autoencoder.ckpt", "autoencoder.encoder"),
        decoder=PartialCheckpoint("vanilla_vae_autoencoder.ckpt", "autoencoder.decoder")
    )

    vae = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=CNN(1, 256, 32, 1, None, 8, 2, True, False, "batchnorm", "relu"),
        decoder=CNN(128, 1, 1, 32, None, 8, 2, False, True, "batchnorm", "relu"),
        prior=GaussianPrior(loss_coeff=0.1),
        checkpoints=checkpoints,
        out_transforms=UnNormalize((0.1307,), (0.3081,))
    )

    # import IPython; IPython.embed(); exit(1)

    vae.setup()
    trainer = Trainer(limit_train_batches=250, limit_val_batches=40, max_epochs=_MAX_EPOCH, enable_progress_bar=True)
    results = trainer.test(vae, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE


if __name__ == "__main__":
    test_vanilla_vae_encoder_decoder()
    test_vanilla_vae_autoencoder()

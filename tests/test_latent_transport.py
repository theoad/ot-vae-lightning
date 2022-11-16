"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a CI

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import torch
from torchvision.transforms.transforms import GaussianBlur
from pytorch_lightning import Trainer, seed_everything
from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from ot_vae_lightning.model import VAE
from ot_vae_lightning.prior import GaussianPrior
from ot_vae_lightning.data import MNIST32
from ot_vae_lightning.networks import AutoEncoder
from ot_vae_lightning.ot import LatentTransport

_PSNR_PERFORMANCE = 15
_MAX_EPOCH = 2


transport_kwargs = dict(
    transformations=GaussianBlur(5, sigma=(1.5, 1.5)),
    transport_type="gaussian",
    transformed_latents_from_train=True,
    make_pd=True,
    verbose=True,
    stochastic=False,
    pg_star=0,
    persistent=True
)


def test_vae_latent_transport(prog_bar=False, batch_size=50):
    seed_everything(42)

    datamodule = MNIST32(train_batch_size=batch_size)

    in_channels, in_resolution = 1, 32  # MNIST32 pads MNIST images such that the resolution is a power of 2
    latent_channels, latent_resolution = 64, 4  # latent vectors will have shape [64, 4, 4]

    autoencoder = AutoEncoder(
        in_channels,
        latent_channels,
        in_resolution,
        latent_resolution,
        capacity=4,
        double_encoded_features=True,
        down_up_sample=True,
        residual=True
    )

    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        autoencoder=autoencoder,
        prior=GaussianPrior(loss_coeff=0.1),
    )

    callbacks = [
        LatentTransport(
            size=model.latent_size,
            transport_dims=(1, 2, 3),
            diag=False,
            logging_prefix="mat_stochastic",
            **transport_kwargs
        ),
        LatentTransport(
            size=model.latent_size,
            transport_dims=(1, 2, 3),
            diag=True,
            logging_prefix="diag_stochastic",
            **transport_kwargs
        ),
        LatentTransport(
            size=model.latent_size,
            transport_dims=(1,),
            diag=False,
            logging_prefix="mat_stochastic_per_needle",
            **transport_kwargs
        ),
        LatentTransport(
            size=model.latent_size,
            transport_dims=(2, 3),
            diag=False,
            logging_prefix="mat_stochastic_per_channel",
            **transport_kwargs
        )
    ]

    # Train
    trainer = Trainer(
        max_epochs=_MAX_EPOCH,
        enable_progress_bar=prog_bar,
        accelerator='auto',
        devices='auto',
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=False
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    for callback in trainer.callbacks:
        if hasattr(callback, 'test_metrics'):
            res = callback.test_metrics.compute()
            assert list(res.values())[0] > _PSNR_PERFORMANCE

    print('VAE CNN Transport test success')


if __name__ == "__main__":
    test_vae_latent_transport(True, 50)

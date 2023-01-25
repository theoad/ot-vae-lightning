"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a CI of latent transport

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
from ot_vae_lightning.data import MNIST32
from ot_vae_lightning.networks import AutoEncoder
from ot_vae_lightning.ot import LatentTransport
from ot_vae_lightning.ot.transport import GMMTransport, DiscreteTransport, GaussianTransport


_PSNR_PERFORMANCE = 14
_MAX_EPOCH = 2

w2_cfg = dict(diag=False, stochastic=False, pg_star=0., make_pd=True, verbose=True, dtype=torch.double)
mixture_cfg = dict(metric='euclidean', p=2., topk=None, temperature=1., training_mode='argmax', inference_mode='argmax')
source_cfg = dict(update_decay=None, update_with_autograd=False, dtype=torch.double)
target_cfg = dict(update_decay=None, update_with_autograd=False, dtype=torch.double)
callback_cfg = dict(
    transformations=GaussianBlur(5, sigma=(1.5, 1.5)),
    source_latents_from_train=False, target_latents_from_train=False, unpaired=True, common_operator=True,
    reset_source=True, store_source=False, reset_target=True, store_target=False,
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
        double_encoded_features=False,
        down_up_sample=True,
        residual="add"
    )

    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        autoencoder=autoencoder,
        prior=None,
    )

    callbacks = [
        LatentTransport(
            size=model.latent_size,
            transport_operator=GaussianTransport,
            transport_dims=(1, 2, 3),
            logging_prefix="gaussian",
            transport_cfg={**w2_cfg, 'diag': False},
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            **callback_cfg
        ),
        LatentTransport(
            size=model.latent_size,
            transport_operator=GMMTransport,
            transport_dims=(1,),
            logging_prefix="gmm",
            transport_type='argmax',
            transport_cfg={**w2_cfg, 'diag': True},  # TODO: fix diag=False
            source_cfg={**source_cfg, 'mixture_cfg': {**mixture_cfg, 'n_components': 10}},
            target_cfg={**target_cfg, 'mixture_cfg': {**mixture_cfg, 'n_components': 10}},
            **callback_cfg
        ),
        LatentTransport(
            size=model.latent_size,
            transport_operator=DiscreteTransport,
            transport_dims=(2,3),
            logging_prefix="discrete",
            transport_type='mean',
            source_cfg={**source_cfg, 'mixture_cfg': {**mixture_cfg, 'training_mode': 'mean', 'n_components': 1024, 'temperature': 1e-2}},
            target_cfg={**target_cfg, 'mixture_cfg': {**mixture_cfg, 'training_mode': 'mean', 'n_components': 1024, 'temperature': 1e-2}},
            **{**callback_cfg}
        )
    ]

    # Train
    trainer = Trainer(
        max_epochs=_MAX_EPOCH,
        enable_progress_bar=prog_bar,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        num_sanity_val_steps=100,
        logger=False
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    for callback in trainer.callbacks:
        if isinstance(callback, LatentTransport) and (
                isinstance(callback.transport_operator, GaussianTransport)
                or isinstance(callback.transport_operator, GMMTransport)
        ):
            res = callback.test_metrics.compute()
            assert list(res.values())[0] > _PSNR_PERFORMANCE

    print('VAE CNN Transport test success')


if __name__ == "__main__":
    test_vae_latent_transport(True, 50)

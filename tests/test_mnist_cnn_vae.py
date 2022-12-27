"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a CI CNN VAE

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import torch
from torchvision.datasets import MNIST as RawMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pytorch_lightning import Trainer, seed_everything
from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from ot_vae_lightning.utils.partial_checkpoint import PartialCheckpoint
from ot_vae_lightning.model import VAE
from ot_vae_lightning.prior import GaussianPrior
from ot_vae_lightning.data import MNIST32
from ot_vae_lightning.networks import CNN, AutoEncoder
from ot_vae_lightning.metrics.fid import FrechetInceptionDistance

_PSNR_PERFORMANCE = 13
_MAX_EPOCH = 1


def test_vae_encoder_decoder_training(prog_bar=False, batch_size=50):
    seed_everything(42)

    trainer = Trainer(
        max_epochs=_MAX_EPOCH,
        enable_progress_bar=prog_bar,
        accelerator='auto',
        devices=1,
        num_sanity_val_steps=0
    )
    datamodule = MNIST32(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size
    )

    in_channels, in_resolution = 1, 32  # MNIST32 pads MNIST images such that the resolution is a power of 2
    latent_channels, latent_resolution = 128, 1  # latent vectors will have shape [128, 1, 1]

    encoder = CNN(  # Simple nn.Module
        in_channels,
        latent_channels * 2,  # must double the number of channels in the encoder to allow re-parametrization trick
        in_resolution,
        latent_resolution,
        capacity=8,
        down_sample=True,
        residual="add"
    )

    decoder = CNN(  # Simple nn.Module
        latent_channels,
        in_channels,
        latent_resolution,
        in_resolution,
        capacity=8,
        up_sample=True,
        residual="add"
    )

    metrics = MetricCollection({
        'psnr': PeakSignalNoiseRatio(),
        'fid': FrechetInceptionDistance(),
    })

    model = VAE(  # LightningModule
        metrics=metrics,
        encoder=encoder,
        decoder=decoder,
        prior=GaussianPrior(loss_coeff=0.1),
    )

    assert model.latent_size == torch.Size((latent_channels, latent_resolution, latent_resolution))

    # Train
    trainer.fit(model, datamodule)
    trainer.save_checkpoint("vanilla_vae_encoder_decoder.ckpt")

    # Test
    model.freeze()
    results = trainer.test(model, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE

    inference("vanilla_vae_encoder_decoder.ckpt")

    print('VAE CNN encoder + decoder MNIST test success')


def test_vae_autoencoder_training(prog_bar=False, batch_size=50):
    seed_everything(42)

    trainer = Trainer(
        max_epochs=_MAX_EPOCH,
        enable_progress_bar=prog_bar,
        accelerator='auto',
        devices=1,
        num_sanity_val_steps=0
    )
    datamodule = MNIST32(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size
    )

    in_channels, in_resolution = 1, 32  # MNIST32 pads MNIST images such that the resolution is a power of 2
    latent_channels, latent_resolution = 128, 1  # latent vectors will have shape [128, 1, 1]

    autoencoder = AutoEncoder(
        in_channels,
        latent_channels,
        in_resolution,
        latent_resolution,
        capacity=8,
        double_encoded_features=True,
        down_up_sample=True,
        residual="add"
    )

    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        autoencoder=autoencoder,
        prior=GaussianPrior(loss_coeff=0.1),
    )

    # Train
    trainer.fit(model, datamodule)
    trainer.save_checkpoint("vanilla_vae_autoencoder.ckpt")

    # Test
    model.freeze()
    results = trainer.test(model, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE

    # Checkpoint loading
    vae = VAE.load_from_checkpoint("vanilla_vae_autoencoder.ckpt")
    vae.freeze()

    vae.test_metrics = MetricCollection({'psnr': PeakSignalNoiseRatio()}, prefix='test/metrics/')
    results = trainer.test(vae, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE

    # Partial checkpoint loading
    trainer.save_checkpoint("vanilla_vae_autoencoder.ckpt", weights_only=True)

    print('VAE CNN autoencoder MNIST test success')

    checkpoints = dict(
        encoder=PartialCheckpoint("vanilla_vae_autoencoder.ckpt", "autoencoder.encoder"),
        decoder=PartialCheckpoint("vanilla_vae_autoencoder.ckpt", "autoencoder.decoder")
    )

    encoder = CNN(  # Simple nn.Module
        in_channels,
        latent_channels * 2,  # must double the number of channels in the encoder to allow re-parametrization trick
        in_resolution,
        latent_resolution,
        capacity=8,
        down_sample=True,
        residual="add"
    )

    decoder = CNN(  # Simple nn.Module
        latent_channels,
        in_channels,
        latent_resolution,
        in_resolution,
        capacity=8,
        up_sample=True,
        residual="add"
    )

    vae = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=encoder,
        decoder=decoder,
        prior=GaussianPrior(loss_coeff=0.1),
        checkpoints=checkpoints,
    )

    vae.setup()
    vae.freeze()
    results = trainer.test(vae, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE

    print('VAE CNN partial loading test success')


def inference(ckpt_path, prog_bar=False, batch_size=50):
    trainer = Trainer(
        max_epochs=_MAX_EPOCH,
        enable_progress_bar=prog_bar,
        accelerator='auto',
        devices=1,
        num_sanity_val_steps=0
    )

    # Inference
    vae = VAE.load_from_checkpoint(ckpt_path)
    vae.freeze()  # put model in eval automatically

    # wraps user methods (forward, encode, decode) with appropriate pre/post-processing:
    # - normalize images before inputting to the model
    # - de-normalize model outputs
    vae.inference = True

    with torch.no_grad():
        x = torch.randn(10, 1, 28, 28)
        z = vae.encode(x)  # pre-processing is done implicitly
        assert z.shape == torch.Size((10, 128, 1, 1))

        samples = vae.sample(batch_size=5)  # post-processing is done implicitly
        assert samples.shape == torch.Size((5, 1, 28, 28))

        x_hat = vae(x)  # pre-processing and post-processing are done implicitly
        assert x_hat.shape == torch.Size((10, 1, 28, 28))

    # Inference in production. No transforms tailored to the pretrained model. Just raw data !
    raw_mnist = RawMNIST(
        "~/.cache",
        train=False,
        transform=T.ToTensor(),
        download=True
    )

    dl = DataLoader(raw_mnist, batch_size=batch_size, shuffle=False)
    predictions = trainer.predict(vae, dl)
    assert predictions[0].shape == torch.Size((batch_size, 1, 28, 28))  # type: ignore[arg-type]

    psnr = PeakSignalNoiseRatio()
    for i, (x, _) in enumerate(dl):
        psnr.update(vae(x), x)
    assert psnr.compute() > _PSNR_PERFORMANCE

    vae.test_metrics = MetricCollection({
        'psnr': PeakSignalNoiseRatio(),
        'fid': FrechetInceptionDistance(),
    }, prefix='test/metrics/')

    results = trainer.test(vae, dl)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE


def test_inference(prog_bar=False, batch_size=50):
    inference("vanilla_vae_encoder_decoder.ckpt", prog_bar, batch_size)
    inference("vanilla_vae_autoencoder.ckpt", prog_bar, batch_size)

    print('VAE CNN inference test success')


if __name__ == "__main__":
    test_vae_encoder_decoder_training(True, 50)
    test_vae_autoencoder_training(True, 50)
    test_inference(True, 50)

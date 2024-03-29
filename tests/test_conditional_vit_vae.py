"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a CI for ViT VAE

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
from ot_vae_lightning.prior import ConditionalGaussianPrior
from ot_vae_lightning.data import CIFAR10
from ot_vae_lightning.networks import ViT
from ot_vae_lightning.ot import LatentTransport, ConditionalLatentTransport
from ot_vae_lightning.data.progressive_callback import ProgressiveTransform, PgTransform
from ot_vae_lightning.ot.transport import GaussianTransport

_PSNR_PERFORMANCE = 18
_TRANSPORT_PERFORMANCE = 18
_MAX_EPOCH = 2
_DIM = 128


def test_vae_vit_training(prog_bar=False, batch_size=50):
    seed_everything(42)

    datamodule = CIFAR10(train_batch_size=batch_size)

    vit_tiny_cfg = dict(
        image_size=32,
        patch_size=8,
        dim=_DIM,
        depth=3,
        heads=4,
        mlp_dim=_DIM * 4,
        channels=3,
        dropout=0.1,
        emb_dropout=0.,
        num_classes=10
    )

    encoder = ViT(
        n_embed_tokens=2,
        n_input_tokens=None,
        output_tokens='embed',
        patch_to_embed=True,
        embed_to_patch=False,
        **vit_tiny_cfg
    )

    decoder = ViT(
        n_embed_tokens=None,
        n_input_tokens=1,
        output_tokens='embed',
        patch_to_embed=False,
        embed_to_patch=True,
        **vit_tiny_cfg
    )

    prior = ConditionalGaussianPrior(
        dim=(1, _DIM),
        num_classes=10,
        loss_coeff=0.1,
        empirical_kl=False,
        reparam_dim=1,
        annealing_steps=1000
    )

    model = VAE(  # LightningModule
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        conditional=True
    )

    assert model.latent_size == torch.Size((1, vit_tiny_cfg['dim']))

    callbacks = [
        ProgressiveTransform(PgTransform(GaussianBlur, {'sigma': [(1, 1), (0.5, 0.5)]}, kernel_size=5), schedule=[0, 1]),
        LatentTransport(
            size=model.latent_size,
            transport_dims=(2,),
            transformations=GaussianBlur(5, sigma=(1.5, 1.5)),
            transport_operator=GaussianTransport,
            logging_prefix="embed_token",
            transport_cfg=dict(diag=False, make_pd=True, verbose=True, stochastic=False, pg_star=0, persistent=True),
            source_latents_from_train=False, target_latents_from_train=True, unpaired=True, common_operator=True,
            reset_source=True, store_source=False, reset_target=False, store_target=False,
            source_cfg=dict(update_decay=None, update_with_autograd=False, dtype=torch.double),
            target_cfg=dict(update_decay=0.995, update_with_autograd=False, dtype=torch.double, reduce_on_update=False),
        ),
    ]

    # Train
    trainer = Trainer(
        max_epochs=_MAX_EPOCH,
        enable_progress_bar=prog_bar,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=False
    )

    trainer.fit(model, datamodule)
    trainer.save_checkpoint("conditional_vit.ckpt")

    # Test
    model.freeze()
    results = trainer.test(model, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE

    print('VAE Conditional ViT CIFAR10 test success')
    print('VAE Progressive Transforms test success')

    for callback in trainer.callbacks:
        if hasattr(callback, 'test_metrics'):
            res = callback.test_metrics.compute()
            assert list(res.values())[0] > _TRANSPORT_PERFORMANCE

    print('VAE ViT Transport test success')


if __name__ == "__main__":
    test_vae_vit_training(True, 50)

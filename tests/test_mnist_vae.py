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
from ot_vae_lightning.networks import CNN
from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio


def test_vanilla_vae():
    seed_everything(42)

    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=CNN(1, 128, 32, 2, None, 4, 2, True, False),
        decoder=CNN(64, 1, 2, 32, None, 4, 2, False, True),
        prior=GaussianPrior()
    )

    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    datamodule = MNISTDatamodule(train_batch_size=40)
    trainer.fit(model, datamodule)

    results = trainer.test(model, datamodule)
    import IPython; IPython.embed(); exit(1)
    assert results[0]['test/psnr'] > 10

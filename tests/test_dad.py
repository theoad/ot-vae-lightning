"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a CI fot Discrete Auto Diffusion (DAD)

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import torch

from pytorch_lightning import Trainer, seed_everything
from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from ot_vae_lightning.model.discrete_auto_diffuser import DAD
from ot_vae_lightning.prior.codebook import CodebookPrior as Vocabulary
from ot_vae_lightning.data import MNIST
from ot_vae_lightning.networks import ViT

_PSNR_PERFORMANCE = 13      # TODO: make this higher
_MAX_EPOCH = 2
_DIM = 64


def test_dad(prog_bar=False, batch_size=32):
    seed_everything(42)

    datamodule = MNIST(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size
    )

    vit_tiny_cfg = dict(
        image_size=28,
        patch_size=7,
        dim=_DIM,
        depth=2,
        heads=4,
        mlp_dim=_DIM * 4,
        channels=1,
        dropout=0.1,
        emb_dropout=0.
    )

    encoder = ViT(
        n_embed_tokens=0,       # <SOS> token
        n_input_tokens=None,
        output_tokens='input',
        patch_to_embed=True,
        embed_to_patch=False,
        **vit_tiny_cfg
    )

    decoder = ViT(
        n_embed_tokens=None,
        n_input_tokens=encoder.total_num_tokens,
        output_tokens='input',
        patch_to_embed=False,
        embed_to_patch=True,
        **vit_tiny_cfg
    )

    autoregressive = ViT(
        n_embed_tokens=0,
        n_input_tokens=encoder.total_num_tokens,
        output_tokens='input',
        patch_to_embed=False,
        embed_to_patch=False,
        causal_mask=True,
        **vit_tiny_cfg
    )

    vocab = Vocabulary(
        num_embeddings=512,
        latent_size=encoder.out_size,
        embed_dims=(2,),
        similarity_metric='l2',
        separate_key_values=False,
        mode='sample',
        temperature=1,
        loss_coeff=1
    )

    vocab_embed = torch.nn.Embedding(vocab.num_embeddings, _DIM)
    decoder_head = torch.nn.Linear(_DIM, vocab_embed.num_embeddings)
    autoregressive = torch.nn.Sequential(vocab_embed, autoregressive, decoder_head)

    vocab_config = dict(
        permute=False
    )

    model = DAD(  # LightningModule
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=encoder,                              # q(z1   | x)
        decoder=decoder,                              # p(x    | z1)
        autoregressive_decoder=autoregressive,        # p(zt-1 | zt)
        vocabulary=vocab,
        ce_coeff=1e-3,
        prior_kwargs=vocab_config,
        learning_rate=1e-3
    )

    assert model.latent_size == torch.Size([(28 // 7) ** 2, vit_tiny_cfg['dim']])

    # Train
    trainer = Trainer(
        max_epochs=_MAX_EPOCH,
        enable_progress_bar=prog_bar,
        accelerator='auto',
        devices='auto',
        logger=False,
    )

    trainer.fit(model, datamodule)
    trainer.save_checkpoint("dad_vit.ckpt")

    # Test
    model.freeze()
    results = trainer.test(model, datamodule)
    assert results[0]['test/metrics/psnr'] > _PSNR_PERFORMANCE

    print('DAD ViT CIFAR10 test success')


if __name__ == "__main__":
    test_dad(True, 50)

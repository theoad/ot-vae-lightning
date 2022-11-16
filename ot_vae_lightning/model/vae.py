"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a VAE

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import warnings
import inspect
import functools
import itertools
from typing import Tuple, Optional, Dict, List, Union, Callable, Any
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.transforms import GaussianBlur

from torchmetrics import MetricCollection

import ot_vae_lightning.data    # noqa: F401
import ot_vae_lightning.utils as utils
import ot_vae_lightning.data.progressive_callback as progressive
from ot_vae_lightning.prior import Prior
from ot_vae_lightning.model.base import VisionModule, VisionCLI
from ot_vae_lightning.utils import Collage
from ot_vae_lightning.ot import LatentTransport
from ot_vae_lightning.data import TorchvisionDatamodule, ProgressiveTransform, PgTransform
from ot_vae_lightning.utils.partial_checkpoint import PartialCheckpoint


class VAE(VisionModule):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a VAE

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """

    Batch = Dict[str, Union[Tensor, Dict]]
    filter_kwargs = utils.FilterKwargs
    filter_kwargs.__init__ = functools.partialmethod(filter_kwargs.__init__, arg_keys='labels')

    # noinspection PyUnusedLocal
    def __init__(
            self,
            prior: Prior,
            *,
            autoencoder: Optional[nn.Module] = None,
            encoder: Optional[nn.Module] = None,
            decoder: Optional[nn.Module] = None,
            metrics: Optional[MetricCollection] = None,
            checkpoints: Optional[Dict[str, PartialCheckpoint]] = None,
            inference_preprocess: Optional[Callable] = None,
            inference_postprocess: Optional[Callable] = None,
            ema_decay: Optional[float] = None,
            learning_rate: float = 1e-3,
            lr_sched_metric: Optional[str] = None,  # If `None`, CosineLr is used
            conditional: bool = False,
            expansion: int = 1,
            prior_kwargs: Dict[str, Any] = {},
    ) -> None:
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

        :param prior: prior class (derived from abstract class Prior)
        :param autoencoder: A nn.Module with methods `self.encode` and `self.decode`.
                            Can be left unfilled if `encoder and `decoder` parameter are filled.
        :param encoder: A nn.Module with method `self.encode`. Can be left unfilled if `autoencoder` is filled
        :param decoder: A nn.Module with method `self.decode`. Can be left unfilled if `autoencoder` is filled
        :param learning_rate: The model learning rate. Default `1e-3`
        :param checkpoints: See father class (ot_vae_lightning.model.base.BaseModule) docstring
        """
        super().__init__(metrics, checkpoints, False, inference_preprocess, inference_postprocess, ema_decay)
        if autoencoder is None and (encoder is None or decoder is None):
            raise ValueError("At least one of `autoencoder` or (`encoder`, `decoder`) parameters must be set")
        if autoencoder is not None and (encoder is not None or decoder is not None):
            raise ValueError("Setting both `autoencoder` and `encoder` or `decoder` is ambiguous")

        self.save_hyperparameters(ignore=['metrics'])

        self.loss = self.nelbo
        self.prior = prior
        self._warn_call('prior.forward'); self._warn_call('prior.sample')
        if autoencoder is not None:
            assert (
                    isinstance(autoencoder, nn.Module) and
                    hasattr(autoencoder, 'encode') and hasattr(autoencoder, 'decode')
            ), 'Parameter `autoencoder` should be a nn.Module and implement the methods `encode` and `decode`'
            self.autoencoder = autoencoder
            self._encode_func = autoencoder.encode; self._warn_call('autoencoder.encode')
            self._decode_func = autoencoder.decode; self._warn_call('autoencoder.decode')
            # Don't set self.encoder and self.decoder in order for checkpoints loading to not be ambiguous
        else:
            assert encoder is not None
            assert decoder is not None
            self.encoder = encoder
            self.decoder = decoder
            self._encode_func = encoder; self._warn_call('encoder.forward')
            self._decode_func = decoder; self._warn_call('decoder.forward')
            # Don't set self.autoencoder in order for checkpoints loading to not be ambiguous

        self._expand = functools.partial(utils.replicate_batch, n=expansion)
        self._reduce_mean = functools.partial(utils.mean_replicated_batch, n=expansion)
        self._reduce_std = functools.partial(utils.std_replicated_batch, n=expansion)
        self.prior_kwargs = prior_kwargs

    @progressive.transform_batch_tv()
    def batch_preprocess(self, batch) -> Batch:
        samples, labels = batch
        kwargs = {'labels': labels} if self.hparams.conditional else {}
        return {
            'samples': samples,
            'target': samples,
            'kwargs': kwargs
        }

    @VisionModule.postprocess
    @VisionModule.preprocess
    def forward(self, samples: Tensor, expand: bool = False, **kwargs) -> Tensor:
        latents = self.encode(samples, expand=expand, no_preprocess_override=True, **kwargs)
        reconstructions = self.decode(latents, expand_kwargs=expand, no_postprocess_override=True, **kwargs)
        return reconstructions

    def optim_parameters(self):
        if hasattr(self, 'autoencoder'):
            return (p for p in itertools.chain(self.autoencoder.parameters(), self.prior.parameters()) if p.requires_grad)
        return (p for p in itertools.chain(self.encoder.parameters(), self.decoder.parameters(), self.prior.parameters()) if p.requires_grad)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.optim_parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4, betas=(0.9, 0.999)
        )
        if self.hparams.lr_sched_metric is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-6, last_epoch=self.trainer.global_step or -1
            )
            lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        else:
            name = self.val_metrics.prefix + self.hparams.lr_sched_metric
            mode = 'max' if self.val_metrics[self.hparams.lr_sched_metric].higher_is_better else 'min'
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=mode, factor=0.75, patience=8, verbose=True, threshold=1e-1, min_lr=1e-6
            )
            lr_scheduler = {"scheduler": scheduler, "monitor": name}
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    def recon_loss(self, reconstructions: Tensor, target: Tensor, **kwargs) -> Tensor:
        return F.l1_loss(reconstructions, target, reduction="none").sum(dim=(1, 2, 3)).mean()

    def prior_loss(self, prior_loss: Tensor, **kwargs) -> Tensor:
        return prior_loss.mean()

    # noinspection PyUnusedLocal
    def nelbo(self, batch: Batch, batch_idx: int) -> Tuple[Tensor, Dict[str, Tensor], Batch]:
        samples, target, kwargs = batch['samples'], batch['target'], batch['kwargs']
        batch_size = samples.size(0)

        latents, prior_loss = self.encode(samples, expand=True, return_prior_loss=True, **kwargs)
        reconstructions = self.decode(latents, expand_kwargs=True, **kwargs)
        reconstructions_mean = self._reduce_mean(reconstructions)

        prior_loss = self.prior_loss(prior_loss, **kwargs) / np.prod(samples.shape[1:])
        recon_loss = self.recon_loss(reconstructions_mean, target, **kwargs) / np.prod(samples.shape[1:])

        loss = recon_loss + prior_loss
        logs = {
            'train/loss/total': loss,
            'train/loss/recon': recon_loss,
            'train/loss/prior': prior_loss
        }

        batch['preds'] = reconstructions[:batch_size]
        batch['latents'] = latents[:batch_size]
        batch['preds_mean'] = reconstructions_mean
        return loss, logs, batch

    @property
    def latent_size(self):
        if hasattr(self, 'autoencoder'):
            enc_out = self.autoencoder.latent_size
        else:
            assert hasattr(self, 'encoder')
            enc_out = self.encoder.out_size
        return self.prior.out_size(enc_out)

    @VisionModule.preprocess
    def encode(
            self,
            samples: Tensor,
            return_prior_loss: bool = False,
            expand: bool = False,
            **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        with self.filter_kwargs(self._encode_func) as encode, self.filter_kwargs(self.prior) as prior:
            encodings = encode(samples, **kwargs)
            if expand: encodings, kwargs = self._expand(encodings), self._expand(kwargs)
            latents, prior_loss = prior(encodings, **kwargs, step=self.global_step, **self.prior_kwargs)

        if return_prior_loss:
            assert not self.inference, 'The prior loss cannot be returned when the model is in inference mode'
            return latents, prior_loss

        return latents

    @VisionModule.postprocess
    def decode(self, latents: Tensor, expand_kwargs: bool = False, **kwargs) -> Tensor:
        if expand_kwargs:
            kwargs = self._expand(kwargs)

        with self.filter_kwargs(self._decode_func) as decode:
            return decode(latents, **kwargs)

    @VisionModule.postprocess
    def sample(self, batch_size: int, **kwargs) -> Tensor:
        with self.filter_kwargs(self.prior.sample) as sample:
            latents = sample((batch_size, *self.latent_size), device=self.device, **kwargs)

        return self.decode(latents, **kwargs, no_postprocess_override=True)

    @Collage.log_method
    def reconstruction(self, batch: Batch) -> List[Tensor]:
        samples, target, kwargs = batch['samples'], batch['target'], batch['kwargs'],
        batch_size = samples.size(0)
        reconstructions = self(samples, expand=True, **kwargs)
        reconstructions_mean = self._reduce_mean(reconstructions)
        reconstructions_std = self._reduce_std(reconstructions)
        realizations = [reconstructions[batch_size * i:batch_size * (i + 1)] for i in range(self.hparams.expansion)]
        return [target, reconstructions_mean, *realizations, reconstructions_std]

    @Collage.log_method
    def generation(self, batch: Batch) -> List[Tensor]:
        samples, kwargs = batch['samples'], batch['kwargs']
        return self.sample(samples.size(0) * 4, **kwargs).chunk(4, dim=0)

    def _warn_call(self, method) -> None:
        module = self
        for attr in method.rsplit('.'):
            module = getattr(module, attr)
        args = inspect.signature(self.filter_kwargs.__init__).parameters['arg_keys'].default
        if isinstance(args, str): args = [args]
        for arg in args:
            if utils.hasarg(module, arg): continue
            if arg == 'labels':
                if self.hparams.conditional: warnings.warn(f"""
                `conditional` is specified but `{method}` doesn't accept `{arg}` parameter
                """)
            else: warnings.warn(f"""
            `{arg}` specified as a key-worded argument but `{method}` doesn't accept `{arg}` parameter.
            """)


class VaeCLI(VisionCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_lightning_class_args(Collage, "collage")
        parser.set_defaults({"collage.log_interval": 100, "collage.num_samples": 8})
        # parser.link_arguments("data.IMG_SIZE", "model.encoder.init_args.image_size", apply_on="instantiate")
        # parser.link_arguments("data.IMG_SIZE", "model.decoder.init_args.image_size", apply_on="instantiate")
        # parser.link_arguments("model.encoder.init_args.dim", "model.decoder.init_args.dim", apply_on="instantiate")


if __name__ == '__main__':
    cli = VaeCLI(
        VAE, TorchvisionDatamodule,
        subclass_mode_model=False,
        subclass_mode_data=True,
        save_config_filename='cli_config.yaml',
        save_config_overwrite=True,
        run=False
    )

    transport_kwargs = dict(
        size=cli.model.latent_size,
        transformations=GaussianBlur(5, sigma=(1.5, 1.5)),
        transport_type="gaussian",
        transformed_latents_from_train=True,
        make_pd=True,
        verbose=True,
        pg_star=0,
    )

    # ProgressiveGaussianBlur = PgTransform(
    #     GaussianBlur, {'sigma': list(zip(np.linspace(10, 0, 50), np.linspace(10, 0, 50)))[:-1]}, kernel_size=3
    # )

    callbacks = [
        # ProgressiveTransform(ProgressiveGaussianBlur, list(range(50))),
        LatentTransport(
            transport_dims=(1,),
            diag=False,
            stochastic=False,
            logging_prefix="mat_per_needle",
            **transport_kwargs
        ),
        LatentTransport(
            transport_dims=(1,),
            diag=True,
            stochastic=False,
            logging_prefix="mat_per_needle_diag",
            **transport_kwargs
        ),
    ]

    cli.trainer.callbacks += callbacks
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)

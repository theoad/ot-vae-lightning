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
from typing import Tuple, Optional, Dict, List, Union, Any, Literal
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ot_vae_lightning.model.base import VisionModule, VisionCLI
import ot_vae_lightning.utils as utils
import ot_vae_lightning.data.progressive_callback as progressive
from ot_vae_lightning.prior import Prior
from ot_vae_lightning.ot import LatentTransport
from ot_vae_lightning.data import TorchvisionDatamodule
from ot_vae_lightning.utils import Collage, FilterKwargs

__all__ = ['VAE']


class VAE(VisionModule):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a VAE

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """

    Batch = Dict[str, Union[Tensor, Dict]]
    filter_kwargs = FilterKwargs
    filter_kwargs.__init__ = functools.partialmethod(FilterKwargs.__init__, arg_keys='labels')

    # noinspection PyUnusedLocal
    def __init__(
            self,
            *base_args,
            monitor: str = "psnr",
            mode: Literal['min', 'max'] = "max",
            prior: Optional[Prior] = None,
            autoencoder: Optional[nn.Module] = None,
            encoder: Optional[nn.Module] = None,
            decoder: Optional[nn.Module] = None,
            conditional: bool = False,
            expansion: int = 1,
            prior_kwargs: Dict[str, Any] = {},
            **base_kwargs
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

        :param prior: prior class (derived from abstract class Prior). If left unfilled, the class will behave like a
                      regular auto-encoder
        :param autoencoder: A nn.Module with methods `self.encode` and `self.decode`.
                            Can be left unfilled if `encoder and `decoder` parameter are filled.
        :param encoder: A nn.Module with method `self.encode`. Can be left unfilled if `autoencoder` is filled
        :param decoder: A nn.Module with method `self.decode`. Can be left unfilled if `autoencoder` is filled
        :param learning_rate: The model learning rate. Default `1e-3`
        """
        super().__init__(*base_args, monitor=monitor, mode=mode, **base_kwargs)
        if autoencoder is None and (encoder is None or decoder is None):
            raise ValueError("At least one of `autoencoder` or (`encoder`, `decoder`) parameters must be set")
        if autoencoder is not None and (encoder is not None or decoder is not None):
            raise ValueError("Setting both `autoencoder` and `encoder` or `decoder` is ambiguous")

        self.save_hyperparameters(ignore=['metrics'])

        self.loss = self.nelbo
        self.prior = prior
        if self.prior is not None: self._warn_call('prior.forward'); self._warn_call('prior.sample')

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
        param_list = [self.autoencoder.parameters()] if hasattr(self, 'autoencoder') else \
            [self.encoder.parameters(), self.decoder.parameters()]
        if self.prior is not None: param_list.append(self.prior.parameters())
        return filter(lambda p: p.requires_grad, itertools.chain(*param_list))

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.optim_parameters(), lr=1e-3, betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=self.mode, factor=0.75, patience=8, verbose=True, threshold=1e-1, min_lr=1e-6
        )
        lr_scheduler = {"scheduler": scheduler, "monitor": self.monitor}
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    def recon_loss(self, reconstructions: Tensor, target: Tensor, **kwargs) -> Tensor:
        return F.mse_loss(reconstructions, target)

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
        recon_loss = self.recon_loss(reconstructions_mean, target, **kwargs)

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
        if self.prior is None: return enc_out
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

            if self.prior is None:
                latents, prior_loss = encodings, torch.zeros(encodings.size(0), out=torch.empty_like(encodings))
            else:
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
        if self.prior is not None:
            with self.filter_kwargs(self.prior.sample) as sample:
                latents = sample((batch_size, *self.latent_size), device=self.device, **kwargs)
        else: latents = torch.randn((batch_size, *self.latent_size), device=self.device)

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


if __name__ == '__main__':
    import re
    import lovely_tensors as lt
    import ot_vae_lightning.data  # noqa: F401
    from ot_vae_lightning.ot.transport import GaussianTransport, GMMTransport

    lt.monkey_patch()
    # torch.set_float32_matmul_precision('high')

    cli = VisionCLI(
        VAE, TorchvisionDatamodule,
        subclass_mode_model=False,
        subclass_mode_data=True,
        save_config_filename='cli_config.yaml',
        save_config_overwrite=True,
        run=False
    )

    transforms = [
        T.GaussianBlur(5, sigma=(2, 2)),
        T.GaussianBlur(9, sigma=(4, 4)),
        T.RandomErasing(p=1., scale=list(np.linspace(0.1, 2, 10)), ratio=list(np.linspace(0.01, 5, 10))),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        T.RandomPerspective(distortion_scale=0.25, p=1.),
        T.RandomRotation(20),
        T.Compose([T.Lambda(lambda t: (255 * t).type(torch.uint8)), T.RandAugment(5, 8), T.Lambda(lambda t: t.float() / 255)])
    ]

    cli.trainer.callbacks += [
        LatentTransport(
            transport_dims=(1,),
            transport_operator=GMMTransport,
            logging_prefix=f"mat_per_needle_{utils.camel2snake(transform.__class__.__name__)}",
            size=cli.model.latent_size,
            transformations=transform,
            source_latents_from_train=False,
            target_latents_from_train=False,
            unpaired=True,
            verbose=True,
            common_operator=True,
            diag=False,
            make_pd=True,
            stochastic=True,
            pg_star=0.,
            n_components_source=16,
            n_components_target=64,
            transport_type='sample'
        ) for transform in transforms
    ]

    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)

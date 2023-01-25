"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a Discrete Auto Diffusion (DAD)

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Union, Dict

import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ot_vae_lightning.data    # noqa: F401
from ot_vae_lightning.model.base import VisionCLI
from ot_vae_lightning.prior.codebook import CodebookPrior
from ot_vae_lightning.model.vae import VAE
from ot_vae_lightning.data.torchvision_datamodule import TorchvisionDatamodule

__all__ = ['DAD', 'DadCLI']


class DAD(VAE):
    """
    `PyTorch <https://pytorch.org/>`_ implementation of a Discrete Auto Diffusion (DAD)

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    # noinspection PyUnusedLocal
    def __init__(
            self,
            *vae_args,
            prior: CodebookPrior,
            autoregressive_decoder: nn.Module,
            ce_coeff: float = 1.,
            **vae_kwargs,
    ) -> None:
        super().__init__(*vae_args, prior=prior, **vae_kwargs)
        self.token_dims = prior.dimensionality
        self.n_tokens = int(np.prod(prior.batch_shape))
        self.num_embeddings = prior.num_embeddings
        self.autoregressive_decoder = autoregressive_decoder

    def prior_loss(self, prior_loss: Tensor, artifacts: Dict[str, Union[Tensor, Categorical]], **kwargs) -> Tensor:
        distributions, indices = artifacts['distribution'], artifacts['indices']
        logits = self.autoregressive_decoder(indices.detach())  # `indices` are samples from `distribution_models`: q(z1 | x)
        labels = distributions.probs    # we want the actual distribution_models q(zt | zt-1) - not the samples
        expected_shape = torch.Size([prior_loss.size(0), self.n_tokens, self.num_embeddings])
        assert labels.shape == logits.shape == expected_shape

        # p(zt-1 | zt), q(zt-1 | zt-2) with shift, so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # \sum_t KL(q(zt-1 | zt-2) || p(zt-1 | zt))
        ce_loss = F.cross_entropy(
            shift_logits.transpose(-1, -2),     # F.cross_entropy expects class dim first: [batch, C, *]
            shift_labels.transpose(-1, -2),     # F.cross_entropy expects class dim first: [batch, C, *]
            reduction='none'
        ).sum(-1)

        loss = prior_loss + self.hparams.ce_coeff * ce_loss
        return super().prior_loss(loss, artifacts, **kwargs)

    @VAE.postprocess
    def sample(self, batch_size: int, **kwargs) -> Tensor:
        # embed_ind = Categorical(self.prior.codebook_model.distribution.probs).sample((batch_size, self.n_tokens))
        embed_ind = torch.randint(
            high=self.num_embeddings,
            size=(batch_size, self.n_tokens),
            device=self.device
        )

        # self.autoregressive_decoder.train()  # TODO: remove
        for i in range(self.n_tokens-1):
            next_token_distribution = Categorical(self.autoregressive_decoder(embed_ind)[:, i].softmax(-1))
            embed_ind[:, i+1] = next_token_distribution.sample()
        # self.autoregressive_decoder.eval()

        one_hot = F.one_hot(embed_ind, self.num_embeddings).type_as(self.prior.codebook_model.codebook)
        latents = one_hot @ self.prior.codebook_model.codebook
        latents = self.prior.unflatten_and_unpermute(latents.transpose(0, 1))
        return self.decode(latents, **kwargs, no_postprocess_override=True)


class DadCLI(VisionCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.link_arguments(
            "data.IMG_SIZE",
            "model.autoregressive_decoder.init_args.image_size",
            apply_on="instantiate"
        )

        parser.link_arguments(
            "model.encoder.init_args.dim",
            "model.autoregressive_decoder.init_args.dim",
            apply_on="instantiate"
        )

        parser.link_arguments(
            "model.encoder.total_num_tokens",
            "model.decoder.init_args.n_input_tokens",
            apply_on="instantiate"
        )

        parser.link_arguments(
            "model.encoder.total_num_tokens",
            "model.autoregressive_decoder.init_args.n_input_tokens",
            apply_on="instantiate"
        )

        parser.link_arguments(
            "model.num_embeddings",
            "model.autoregressive_decoder.init_args.vocab_size",
            apply_on="instantiate"
        )

        parser.link_arguments(
            "model.encoder.out_size",
            "model.prior.init_args.latent_size",
            apply_on="instantiate"
        )


if __name__ == '__main__':
    cli = DadCLI(
        DAD, TorchvisionDatamodule,
        subclass_mode_model=False,
        subclass_mode_data=True,
        save_config_filename='cli_config.yaml',
        save_config_overwrite=True,
        run=True
    )

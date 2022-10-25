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
from ot_vae_lightning.prior.codebook import CodebookPrior as Vocabulary
from ot_vae_lightning.model.vae import VAE, VaeCLI
from ot_vae_lightning.data import TorchvisionDatamodule


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
            vocabulary: Vocabulary,
            autoregressive_decoder: nn.Module,
            ce_coeff: float = 1.,
            **vae_kwargs,
    ) -> None:
        super().__init__(
            prior=vocabulary,
            **vae_kwargs
        )
        self.token_dims = set(vocabulary.all_dims).difference(set(vocabulary.embed_dims))
        self.n_tokens = int(np.prod([self.latent_size[dim-1] for dim in self.token_dims]))
        self.autoregressive_decoder = autoregressive_decoder
        self.prior_kwargs['return_distributions'] = True
        self.prior_kwargs['return_indices'] = True

    @property
    def vocabulary(self) -> Vocabulary:
        return self.prior   # type: ignore[return-type]

    def prior_loss(self, artifacts: Dict[str, Union[Tensor, Categorical]], **kwargs) -> Tensor:
        vocab_loss, distributions, indices = artifacts['loss'], artifacts['distributions'], artifacts['indices']
        logits = self.autoregressive_decoder(indices.detach())  # `indices` are samples from `distributions`: q(z1 | x)
        labels = distributions.probs    # we want the actual distributions q(zt | zt-1) - not the samples
        expected_shape = torch.Size([vocab_loss.size(0), self.n_tokens, self.vocabulary.num_embeddings])
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

        loss = vocab_loss + self.hparams.ce_coeff * ce_loss
        return super().prior_loss(loss)

    @VAE.postprocess
    def sample(self, batch_size: int, **kwargs) -> Tensor:
        embed_ind = torch.randint(
            high=self.vocabulary.num_embeddings,
            size=(batch_size, self.n_tokens),
            device=self.device
        )

        self.autoregressive_decoder.train()  # TODO: remove
        for i in range(self.n_tokens-1):
            next_token_distribution = Categorical(self.autoregressive_decoder(embed_ind)[:, i].softmax(-1))
            embed_ind[:, i+1] = next_token_distribution.sample()
        self.autoregressive_decoder.eval()

        latents = self.vocabulary.values(embed_ind)
        # latents = utils.unflatten_and_unpermute(latents, self.latent_size, self.vocabulary.embed_dims)
        return self.decode(latents, **kwargs, no_postprocess_override=True)


class DadCLI(VaeCLI):
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
            "model.vocabulary.num_embeddings",
            "model.autoregressive_decoder.init_args.vocab_size",
            apply_on="instantiate"
        )

        parser.link_arguments(
            "model.encoder.out_size",
            "model.vocabulary.init_args.latent_size",
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

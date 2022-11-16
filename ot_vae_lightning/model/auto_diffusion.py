from typing import Optional, List, Union
import functools
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
import ot_vae_lightning.utils as utils
from ot_vae_lightning.model.base import VisionModule
from ot_vae_lightning.model.vae import VAE, VaeCLI
from ot_vae_lightning.utils import Collage
from ot_vae_lightning.data import TorchvisionDatamodule


class AutoDiffusion(VAE):
    n_steps = int(1e1)
    filter_kwargs = utils.FilterKwargs
    filter_kwargs.__init__ = functools.partialmethod(filter_kwargs.__init__, arg_keys=['labels', 'time'])

    def batch_preprocess(self, batch) -> VAE.Batch:
        pbatch = super().batch_preprocess(batch)
        batch_size = pbatch['samples'].size(0)
        t = torch.rand(batch_size, device=self.device)
        t = 0.5 * torch.tanh(10 * (t-0.5)) + 0.5
        pbatch['kwargs']['time'] = t
        return pbatch

    def prior_loss(self, prior_loss: Tensor, **kwargs) -> Tensor:
        t = kwargs['time']
        # beta_t = 0.5 * torch.tanh(10 * (t-0.5)) + 0.5
        return (t * prior_loss).mean()

    @VisionModule.postprocess
    def sample(
            self,
            batch_size: int,
            steps: Optional[List[float]] = None,
            improved_algorithm: bool = False,
            **kwargs
    ) -> Union[Tensor, List[Tensor]]:
        x_hat, ones, intermediate = None, torch.ones_like(kwargs['time']), []
        with self.filter_kwargs(self.prior.sample) as sample:
            xs = sample((batch_size, *self.latent_size), device=self.device, **{**kwargs, 'time': ones})

        step_size = 1 / self.n_steps
        for i, s in enumerate(np.linspace(1, step_size, self.n_steps)):
            x_hat = self.decode(xs, **{**kwargs, 'time': ones * s}, no_postprocess_override=True)
            if improved_algorithm:
                xs -= (
                        self.encode(x_hat, **{**kwargs, 'time': ones * (s - step_size)}, no_preprocess_override=True) -
                        self.encode(x_hat, **{**kwargs, 'time': ones * s}, no_preprocess_override=True)
                )
            else: xs = self.encode(x_hat, **{**kwargs, 'time': ones * (s - step_size)}, no_preprocess_override=True)
            if steps is not None and i in steps: intermediate.append(x_hat)

        return x_hat if steps is None else intermediate

    @Collage.log_method
    def reconstruction(self, batch: VAE.Batch) -> List[Tensor]:
        samples, target, kwargs = batch['samples'], batch['target'], batch['kwargs']
        ones = torch.ones_like(kwargs['time'])
        return [self(samples, **{**kwargs, 'time': ones * t}) for t in np.linspace(0, 1, 10)] + [target]

    @Collage.log_method
    def generation(self, batch: VAE.Batch) -> List[Tensor]:
        samples, kwargs = batch['samples'], batch['kwargs']
        return self.sample(
            samples.size(0),
            steps=[int(i) for i in np.linspace(0, self.n_steps, 10)],
            improved_algorithm=False,
            **kwargs
        )

    @Collage.log_method
    def generation_improved(self, batch: VAE.Batch) -> List[Tensor]:
        samples, kwargs = batch['samples'], batch['kwargs']
        return self.sample(
            samples.size(0),
            steps=[int(i) for i in np.linspace(0, self.n_steps, 10)],
            improved_algorithm=True,
            **kwargs
        )


if __name__ == '__main__':
    cli = VaeCLI(
        AutoDiffusion, TorchvisionDatamodule,
        subclass_mode_model=False,
        subclass_mode_data=True,
        save_config_filename='cli_config.yaml',
        save_config_overwrite=True,
        run=True
    )

"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of an image to image transport callback

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Optional, Any, Tuple, Dict, List
from itertools import accumulate
from operator import mul

from numpy.random import permutation

import torch
from torch import Tensor
from torch.types import _size

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities import rank_zero_warn, move_data_to_device
from pytorch_lightning import Callback

from ot_vae_lightning.ot import GaussianTransport
from ot_vae_lightning.utils.collage import Collage


class LatentTransport(Callback):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a latent image to image transport callback
    This implementation assumes the pytorch-lightning module which is trained has a method named `encode` and a method
    named `decode` and the method `training_step` returns a dictionary where the input images are stored with the
    `samples_key` (argument passed to __init__) and optionally, the latents of these samples are stored with the
    `latents_key` (argument passed to __init__).

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    _CLS_TABLE = {
        "gaussian": GaussianTransport
    }

    TRANSPORT_CHOICE = _CLS_TABLE.keys()

    def __init__(
            self,
            size: _size,
            transformations: callable,
            transport_type: str = "gaussian",
            transport_dims: _size = (1, 2, 3),
            diag: bool = False,
            pg_star: float = 0.,
            stochastic: bool = False,
            persistent: bool = True,
            samples_key: str = 'samples',
            latents_key: str = 'latents',
            logging_prefix: Optional[str] = None,
            transformed_latents_from_train: bool = False,
            num_samples_to_log: int = 8,
            make_pd: bool = False,
            verbose: bool = False,
            class_idx: Optional[int] = None,
            conditional_key: str = 'y'
    ) -> None:
        r"""
        :param size: The size of the Tensors to transport
        :param transformations: The transformations on which the transport will be tested.
        :param transport_type: The transport operator type. Choices are `LatentTransport.TRANSPORT_CHOICE`.
        :param transport_dims: The dimension on which to apply the transport. Examples
         If (1, 2, 3), will transport the input vectors as a whole.
         If (1,), will transport each needle (pixel) individually.
         if (2, 3), will transport each channel individually.
        :param diag: If ``True`` will suppose the samples come from isotropic distributions (with diagonal covariance)
        :param pg_star: Perception-distortion ratio. Must be a float in [0, 1] can be seen as inverse temperature.
        :param stochastic: If ``True`` will add stochasticity to the output (if applicable).
        :param persistent: whether the state buffers (transport operators) should be saved when checkpointing.
        :param samples_key: The key in which are stored the image samples on which to test the transport, in the
         dictionary returned from the `training_step` of the pytorch-lightning module jointly trained with this callback
        :param latents_key: The key in which are stored the latent representation of the image samples on which to test
         the transport, in the dictionary returned from the `training_step` of the pytorch-lightning module jointly
         trained with this callback.
        :param logging_prefix: The logging prefix where to log the results of the transport experiments.
        :param transformed_latents_from_train: If ``True`` uses the latent representation of transformed training
         samples to update the transport operators. This is unadvised since it will slow down training and bias the
         transport experiment (as source and target samples will not come from unpaired distributions)
        :param num_samples_to_log: Number of image samples to log.
        :param verbose: If ``True``, warns about correction value added to the diagonal of the matrices
        :param make_pd: If ``True``, corrects matrices needed to be PD or PSD by adding their minimum eigenvalue to
                        their diagonal.
        """
        super().__init__()
        if transport_type not in LatentTransport.TRANSPORT_CHOICE:
            raise NotImplementedError(f"""
            `transport_type` {transport_type} not supported. Choose one of {LatentTransport.TRANSPORT_CHOICE}
            """)
        all_dims = list(range(1, len(size) + 1))
        if not set(transport_dims).issubset(all_dims):
            raise ValueError(f"""
            Given `size`={size}. The inputs will have the {len(size) + 1} dimensions (with the batch dimensions). 
            Therefore `transport_dims` must be a subset of {all_dims}
            """)
        self.size = size
        self.transformations = transformations
        self.transport_dims = transport_dims
        self.operator_dim = list(set(all_dims).difference(set(self.transport_dims)))
        self.dim = list(accumulate([size[i - 1] for i in self.transport_dims], mul))[-1]
        self.num_operators = list(accumulate([size[i-1] for i in self.operator_dim], mul))[-1]\
            if len(self.operator_dim) > 0 else 1
        self.transformed_latents_from_train = transformed_latents_from_train

        self.samples_key = samples_key
        self.latents_key = latents_key
        self.logging_prefix = f'transport/{transport_type}/{logging_prefix}/'
        self.num_samples_to_log = num_samples_to_log
        self.make_pd = make_pd
        self.verbose = verbose
        self.class_idx = class_idx
        self.conditional_key = conditional_key

        self.val_metrics = None
        self.test_metrics = None

        self.transport_operators = torch.nn.ModuleList([
            LatentTransport._CLS_TABLE[transport_type](
                self.dim,
                diag=diag,
                pg_star=pg_star,
                stochastic=stochastic,
                persistent=persistent,
                make_pd=make_pd,
                verbose=verbose
            ) for _ in range(self.num_operators)
        ])

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_metrics = pl_module.val_metrics.clone(prefix=self.logging_prefix) if self.transformed_latents_from_train else None
        self.test_metrics = pl_module.test_metrics.clone(prefix=self.logging_prefix)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for operator in self.transport_operators:
            operator.reset()

    @torch.no_grad()
    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        samples, kwargs = self._get_samples(pl_module, outputs)

        if self.latents_key in outputs.keys():
            self._update_transport_operators(outputs[self.latents_key], source=False)
        else:
            if self.verbose:
                rank_zero_warn(f"""
                Usage of the {LatentTransport.__class__.__name__} callback will try to learn the distribution of the 
                latent variables emitted by {pl_module.__class__.__name__} during training. The training latents are 
                expected to be found the dictionary outputted by the `training_step` with the key {self.latents_key}. 
                Since this key was not found, the callback will use the `encode` method of 
                {pl_module.__class__.__name__} to compute the latents which might not be the desired behaviour and 
                result in a performance hit. To silence this warning, pass `verbose=False` to the constructor 
                of the callback.
                """)
            pl_module.eval()
            self._update_transport_operators(self._encode(pl_module, samples, **kwargs), source=False)

        if self.transformed_latents_from_train:
            if self.verbose:
                rank_zero_warn(f"""
                The callback will use the `encode` method of {pl_module.__class__.__name__} to compute the latents of 
                the transformed distribution. This is unadvised since it will slow down training and bias the transport 
                experiment (as source and target samples will not come from unpaired distributions). To silence this 
                warning, pass `verbose=False` to the constructor of the callback.
                """)
            pl_module.eval()
            self._update_transport_operators(self._encode(pl_module, self.transformations(samples), **kwargs), source=True)

        pl_module.train()

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.transformed_latents_from_train:
            if self.val_metrics is not None:
                samples, kwargs = self._get_samples(pl_module, outputs)
                latents = self._encode(pl_module, self.transformations(samples), **kwargs)
                samples_transported = self._decode(pl_module, self._transport(latents), **kwargs)
                self.val_metrics.update(samples_transported, samples)
            return

        samples, kwargs = self._get_samples(pl_module, outputs)
        self._update_transport_operators(self._encode(pl_module, self.transformations(samples), **kwargs), source=True)

    @torch.no_grad()
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.test_metrics is None:
            return

        samples, kwargs = self._get_samples(pl_module, outputs)
        latents = self._encode(pl_module, self.transformations(samples), **kwargs)
        samples_transported = self._decode(pl_module, self._transport(latents), **kwargs)
        self.test_metrics(samples_transported, samples)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            return

        # Compute the transport cost distance
        mean_dist = self._val_dist()
        pl_module.log(self.logging_prefix + 'avg_transport_cost', mean_dist)

        if self.num_samples_to_log <= 0:
            return

        collage = self._collage(trainer, pl_module)
        if hasattr(trainer.logger, 'log_image'):
            trainer.logger.log_image(self.logging_prefix, [collage], trainer.global_step)  # type: ignore[arg-type]

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.test_metrics is not None:
            pl_module.log_dict(self.test_metrics.compute())

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.val_metrics is not None:
            self.val_metrics.reset()

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.test_metrics is not None:
            self.test_metrics.reset()

    def _update_transport_operators(self, latents: Tensor, source: bool) -> None:
        latents_rearranged = latents.permute(*self.operator_dim, 0, *self.transport_dims)
        if len(self.operator_dim) > 0:
            latents_rearranged = latents_rearranged.flatten(0, len(self.operator_dim) - 1)
        else:
            latents_rearranged = latents_rearranged.unsqueeze(0)
        key = 'source_samples' if source else 'target_samples'
        for latent, transport_operator in zip(latents_rearranged, self.transport_operators):
            transport_operator.update(**{key: latent.to(transport_operator.device)})

    def _get_samples(self, pl_module: "pl.LightningModule", outputs: STEP_OUTPUT) -> Tuple[Tensor, Dict]:
        if not isinstance(outputs, dict):
            raise ValueError(f"""
            Usage of the {LatentTransport.__class__.__name__} callback demands that the pl_module which is training 
            returns a dictionary in it's `training_step`. The module currently training returned a {type(outputs)}.
            """)

        if self.samples_key not in outputs.keys():
            raise ValueError(f"""
            Usage of the {LatentTransport.__class__.__name__} callback demands that the pl_module which is training 
            returns a dictionary in it's `training_step`, containing the key '{self.latents_key}' in which the image 
            samples' latent representation are located. Since no such key was found, the callback tries to compute 
            the latent representation using the pl_module `encode` method and the image samples located at the key
            '{self.samples_key}'. Fatal: the key '{self.samples_key}' was not found in the output dictionary of the
            pl_module's training_step.
            """)

        outputs = move_data_to_device(outputs, pl_module.device)
        samples = outputs[self.samples_key]
        kwargs = outputs['kwargs'] if 'kwargs' in outputs.keys() else {}

        if self.class_idx is None:
            return samples, kwargs

        if self.conditional_key in outputs:
            condition = outputs[self.conditional_key]
        else:
            if kwargs is None or not isinstance(kwargs, dict) or self.conditional_key not in kwargs.keys():
                raise ValueError(f"""
                `class_idx` was specified in the callback constructor. To get the condition, the callback will search in
                the dictionary returned from the `training_step`, the key '{self.conditional_key}' or the key 'kwargs'
                (which itself is a dictionary that would contain '{self.conditional_key}').
                Fatal: condition not found.
                """)
            condition = kwargs[self.conditional_key]

        filtered_indices = condition == self.class_idx
        samples = samples[filtered_indices]
        kwargs[self.conditional_key] = condition[filtered_indices]
        return samples, kwargs

    def _encode(self, pl_module: "pl.LightningModule", image: Tensor, **kwargs) -> Tensor:
        if not hasattr(pl_module, 'encode'):
            raise NotImplementedError(f"""
            Usage of the {LatentTransport.__class__.__name__} callback demands that the pl_module which is training 
            implements an `encode` method which expects a torch.Tensor image and returns a torch.Tensor latent 
            representation.
            """)

        return pl_module.encode(image, **kwargs)

    def _decode(self, pl_module: "pl.LightningModule", latents: Tensor, **kwargs) -> Tensor:
        if not hasattr(pl_module, 'decode'):
            raise NotImplementedError(f"""
            Usage of the {LatentTransport.__class__.__name__} callback demands that the pl_module which is training 
            implements a `decode` method which expects a torch.Tensor latent representation and returns a torch.Tensor 
            image.
            """)

        return pl_module.decode(latents, **kwargs)

    def _transport(self, latents: Tensor) -> Tensor:
        permutation_map = list(range(latents.dim()))
        permutation_map[0] = len(self.operator_dim)
        for dim in range(1, latents.dim()):
            if dim in self.operator_dim:
                permutation_map[dim] = self.operator_dim.index(dim)
            else:
                assert dim in self.transport_dims, f"Flow assertion error !!!"
                permutation_map[dim] = len(self.operator_dim) + 1 + self.transport_dims.index(dim)

        if len(self.operator_dim) > 0:
            latents_rearranged = latents.permute(*self.operator_dim, 0, *self.transport_dims)
            latents_rearranged = latents_rearranged.flatten(0, len(self.operator_dim) - 1)
        else:
            latents_rearranged = latents.unsqueeze(0)
        latents_transported = torch.stack([
            transport_operator.transport(latent)
            for latent, transport_operator in zip(latents_rearranged, self.transport_operators)
        ])

        if len(self.operator_dim) > 0:
            latents_transported = latents_transported.unflatten(0, [self.size[d - 1] for d in self.operator_dim])  # type: ignore[arg-type]
        else:
            latents_transported = latents_transported.squeeze(0)

        latents_transported = latents_transported.permute(*permutation_map)
        assert latents_transported.shape == latents.shape, "Tensor shape assertion error !!!"

        return latents_transported

    def _collage(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> List[Tensor]:
        # Get validation samples from the target (real) distribution
        batch = pl_module.batch_preprocess(next(iter(trainer.val_dataloaders[0])))
        samples, kwargs = self._get_samples(pl_module, batch)

        # Get the samples from the source (transformed) distribution
        transformed = self.transformations(samples)

        latents = self._encode(pl_module, transformed, **kwargs)
        transformed_decoded = self._decode(pl_module, latents, **kwargs)
        samples_transported = self._decode(pl_module, self._transport(latents), **kwargs)
        img_list = [transformed, transformed_decoded, samples_transported, samples]

        collage = Collage.list_to_collage(img_list, min(samples.shape[0], self.num_samples_to_log))
        return collage

    def _val_dist(self):
        mean_dist = sum([t.compute() for t in self.transport_operators]) / len(self.transport_operators)
        return mean_dist


class ConditionalLatentTransport(Callback):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of a latent image to image conditional
    latent transport callback. The pre-requisite for the pl_module trained jointly with this callback are the same
    as for `LatentTransport`, with the addition of access to a 1-dim condition Tensor either in the output directory
    of the training step or in the `kwargs` key of the outputs.

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """
    def __init__(self, num_classes: int, log_prefix: str, num_samples_to_log: int = 10, *args, **kwargs):
        self.num_classes = num_classes
        self.num_samples_to_log = num_samples_to_log
        self.log_prefix = log_prefix
        self.transports = [
            LatentTransport(*args, **kwargs, class_idx=i, num_samples_to_log=max(1, num_samples_to_log//num_classes))
            for i in range(num_classes)
        ]

    def on_fit_start(self, *args, **kwargs):
        [t.on_fit_start(*args, **kwargs) for t in self.transports]

    def on_train_epoch_start(self, *args, **kwargs):
        [t.on_train_epoch_start(*args, **kwargs) for t in self.transports]

    def on_train_batch_end(self, *args, **kwargs):
        [t.on_train_batch_end(*args, **kwargs) for t in self.transports]

    def on_validation_batch_end(self, *args, **kwargs):
        [t.on_validation_batch_end(*args, **kwargs) for t in self.transports]

    def on_test_batch_end(self, *args, **kwargs):
        [t.on_test_batch_end(*args, **kwargs) for t in self.transports]

    def on_validation_epoch_start(self, *args, **kwargs):
        [t.on_test_batch_end(*args, **kwargs) for t in self.transports]

    def on_test_epoch_start(self, *args, **kwargs):
        [t.on_test_batch_end(*args, **kwargs) for t in self.transports]

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.sanity_checking:
            return

        avg_dist = sum([t._val_dist() for t in self.transports]) / self.num_classes
        pl_module.log(self.log_prefix + 'avg_transport_cost', avg_dist)

        if self.num_samples_to_log <= 0:
            return

        collage = []

        while len(collage) < self.num_samples_to_log:
            collage_indices = permutation(list(range(self.num_classes)))[:self.num_samples_to_log]
            for i in collage_indices:
                collage.append(self.transports[i]._collage(trainer, pl_module))

        collage = Collage.list_to_collage(collage, self.num_samples_to_log)

        if hasattr(trainer.logger, 'log_image'):
            trainer.logger.log_image(self.log_prefix, [collage], trainer.global_step)

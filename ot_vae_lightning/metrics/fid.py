"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of
`Frechet Inception Distance -- FID <https://arxiv.org/abs/1706.08500>`_ inspired from
`torchmetrics' implementation <https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/image/fid.py>`_

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ in collaboration with
`Guy Ohayon <https://github.com/ohayonguy>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from collections import OrderedDict
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
from torchmetrics.image.fid import NoTrainInceptionV3, _compute_fid
from ot_vae_lightning.ot.w2_utils import mean_cov


class NoTrainInceptionV3NoStateDict(NoTrainInceptionV3):
    def state_dict(self, *, destination, prefix, keep_vars):  # type: ignore
        return destination or OrderedDict()


class FrechetInceptionDistance(Metric):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of
    `Frechet Inception Distance -- FID <https://arxiv.org/abs/1706.08500>`_ inspired from
    `torchmetrics' implementation <https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/image/fid.py>`_

    Advantages of this implementation:
        -   Doesn't store all the extracted features in a buffer (compute the mean and covariance on the fly).
        -   Unified API with update receiving `pred`, `target` (allows to have all metrics within one MetricCollection)
        -   Computes the reference (real data) once at the beginning of fit and reuses the real_mean and real_cov throughout training

    Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ in collaboration with
    `Guy Ohayon <https://github.com/ohayonguy>`_

    .. warning:: Work in progress. This implementation is still being verified.

    .. _TheoA: https://github.com/theoad
    """

    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
            self,
            net: Optional[torch.nn.Module] = None,  # inception v3 by default
            feature_size: Optional[int] = 2048,
            to_255: Optional[bool] = False,
            data_range: Optional[Tuple[float, float]] = (0., 1.),
            compute_on_step: Optional[bool] = False,
            dist_sync_on_step: Optional[bool] = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.data_range = data_range[1] - data_range[0]
        self.data_low = data_range[0]
        if net is None:
            valid_int_input = [64, 192, 768, 2048]
            if feature_size not in valid_int_input:
                raise ValueError(
                    f'Integer input to argument `feature` must be one of {valid_int_input},'
                    f' but got {feature_size}.'
                )

            self.net = NoTrainInceptionV3NoStateDict(name='inception-v3-compat', features_list=[str(feature_size)])
            self.to_255 = data_range != (0., 255.)
        else:
            self.net = net
            self.net.eval()
            self.to_255 = to_255

        self.add_state("real_sum", torch.zeros(feature_size, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("real_correlation", torch.zeros(feature_size, feature_size, dtype=torch.double),
                       dist_reduce_fx="sum")
        self.add_state("fake_sum", torch.zeros(feature_size, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("fake_correlation", torch.zeros(feature_size, feature_size, dtype=torch.double),
                       dist_reduce_fx="sum")
        self.add_state("num_real_obs", torch.zeros(1).long(), dist_reduce_fx="sum")
        self.add_state("num_fake_obs", torch.zeros(1).long(), dist_reduce_fx="sum")
        # self.real_mean, self.real_cov = None, None
        # self.real_prepared = False
        rank_zero_info('FID prepared successfully')

    # def _compute_real_statistics(self) -> None:
    #     if self.real_prepared: return
    #     self.real_mean, self.real_cov = compute_mean_cov(self.real_sum, self.real_correlation, self.num_real_obs)
    #     self._persistent['real_mean'] = True
    #     self._persistent['real_cov'] = True
    #     self.real_prepared = True

    def _extract_features(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        if img.size(1) == 1: img = torch.cat([img, img, img], dim=1)
        if self.to_255: img = (255 * (img - self.data_low) / self.data_range).type(torch.uint8)
        features = self.net(img).double().view(img.shape[0], -1)
        sum_features = features.sum(dim=0)
        correlation = features.T @ features
        return sum_features, correlation

    # @rank_zero_only
    # @torch.no_grad()
    # def prepare_metric(self, pl_module: LightningModule) -> None:
    #     self.reset()
    #     dataloader = pl_module.trainer.datamodule.train_dataloader()
    #     for idx, (img, label) in enumerate(dataloader):
    #         self.update(target=img.to(pl_module.device))
    #     self._compute_real_statistics()
    #     rank_zero_info('FID prepared successfully')

    def update(self, generated: Optional[Tensor] = None, samples: Optional[Tensor] = None) -> None:  # type: ignore
        """ Update the state with extracted features
        Args:
            generated: tensor of ``fake`` images
            samples: tensor of ``real`` images
        """
        if generated is not None:
            sum_features, correlation = self._extract_features(generated)
            self.real_sum += sum_features
            self.real_correlation += correlation
            self.num_real_obs += generated.shape[0]
        if samples is not None:
            sum_features, correlation = self._extract_features(samples)
            self.fake_sum += sum_features
            self.fake_correlation += correlation
            self.num_fake_obs += samples.shape[0]

    def compute(self) -> Tensor:
        """ Calculate FID score based on accumulated extracted features from the two distributions """
        if self.num_fake_obs < 1e3 or self.num_real_obs < 1000: return torch.ones(1) * float('inf')
        real_mean, real_cov = mean_cov(self.real_sum, self.real_correlation, self.num_real_obs)
        fake_mean, fake_cov = mean_cov(self.fake_sum, self.fake_correlation, self.num_fake_obs)
        # compute fid
        return _compute_fid(real_mean, real_cov, fake_mean, fake_cov)

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

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import LightningModule
from torchmetrics.image.fid import NoTrainInceptionV3, _compute_fid


def _compute_mean_cov(sum, sum_corr, num_obs):
    """Empirical computation of mean and covariance matrix"""
    mean = sum / num_obs
    cov = (1.0 / (num_obs - 1.0)) * sum_corr - (num_obs / (num_obs - 1.0)) * mean.unsqueeze(1).mm(mean.unsqueeze(0))
    return mean, cov


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

    def __init__(
            self,
            net: Optional[torch.nn.Module] = None,  # inception v3 by default
            feature_size: Optional[int] = 2048,
            to_255: Optional[bool] = False,
            data_range: Optional[Tuple[float, float]] = (0.0, 1.0),
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

            self.net = NoTrainInceptionV3(name='inception-v3-compat', features_list=[str(feature_size)])
            self.to_255 = data_range != (0, 255)
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
        self.add_state("num_real_obs", torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("num_fake_obs", torch.zeros(1), dist_reduce_fx="sum")
        self.real_mean, self.real_cov = None, None
        self.real_prepared = False

    @rank_zero_only
    def prepare_metric(self, pl_module: LightningModule):
        self.reset()
        dataloader = pl_module.train_dataloader()
        with torch.no_grad():
            for idx, (img, label) in enumerate(dataloader):
                self.update(img.to(pl_module.device), real=True)
        self.real_mean, self.real_cov = _compute_mean_cov(self.real_sum, self.real_correlation, self.num_real_obs)
        self._persistent['real_mean'] = True
        self._persistent['real_cov'] = True
        self.real_prepared = True
        rank_zero_info('FID prepared successfully')

    def update(self, img: Tensor, target: Optional[Tensor] = None,
               real: Optional[bool] = False) -> None:  # type: ignore
        """ Update the state with extracted features
        Args:
            img: tensor with images feed to the feature extractor
            target: Unused. for compatibility with other metrics only
            real: bool indicating if imgs belong to the real or the fake distribution
        """
        if self.real_prepared and real: pass
        if self.to_255: img = (255 * (img - self.data_low) / self.data_range).type(torch.uint8)
        features = self.net(img).double().view(img.shape[0], -1)
        sum_features = features.sum(dim=0)
        correlation = features.t().mm(features)
        if real:
            self.real_sum += sum_features
            self.real_correlation += correlation
            self.num_real_obs += features.shape[0]
        else:
            self.fake_sum += sum_features
            self.fake_correlation += correlation
            self.num_fake_obs += features.shape[0]

    def compute(self) -> Tensor:
        """ Calculate FID score based on accumulated extracted features from the two distributions """
        fake_mean, fake_cov = _compute_mean_cov(self.fake_sum, self.fake_correlation, self.num_fake_obs)
        # compute fid
        return _compute_fid(self.real_mean, self.real_cov, fake_mean, fake_cov)

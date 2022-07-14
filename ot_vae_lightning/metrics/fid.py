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
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import LightningModule
from torchmetrics.image.psnr import PeakSignalNoiseRatio

class FID(Metric):
    """
    `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of
    `Frechet Inception Distance -- FID <https://arxiv.org/abs/1706.08500>`_ inspired from
    `torchmetrics' implementation <https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/image/fid.py>`_

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
        self.real_mean, self.real_cov = compute_mean_cov(self.real_sum, self.real_correlation, self.num_real_obs)
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
        fake_mean, fake_cov = compute_mean_cov(self.fake_sum, self.fake_correlation, self.num_fake_obs)
        # compute fid
        return _compute_fid(self.real_mean, self.real_cov, fake_mean, fake_cov)


class NoTrainInceptionV3(FeatureExtractorInceptionV3):
    """FeatureExtractorInceptionV3 always in test mode"""
    def __init__(
        self,
        name: str,
        features_list: List[str],
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> 'NoTrainInceptionV3':
        """ the inception network should not be able to be switched away from evaluation mode """
        return super().train(False)

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        return out[0].reshape(x.shape[0], -1)


class MatrixSquareRoot(Function):
    """
    Square root of a positive definite matrix.
    All credit to:
        https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:
        import scipy

        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        import scipy
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def _compute_fid(
    mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    r"""
    Adjusted version of
        https://github.com/photosynthesis-team/piq/blob/master/piq/fid.py
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular
    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        rank_zero_info(
            f'FID calculation produces singular product; adding {eps} to diagonal of '
            'covaraince estimates'
        )
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


def compute_mean_cov(sum, sum_corr, num_obs):
    """Empirical computation of mean and covariance matrix"""
    mean = sum / num_obs
    cov = (1.0 / (num_obs - 1.0)) * sum_corr - (num_obs / (num_obs - 1.0)) * mean.unsqueeze(1).mm(mean.unsqueeze(0))
    return mean, cov

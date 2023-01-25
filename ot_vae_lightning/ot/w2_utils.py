"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of Wasserstein 2 utilities

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from math import sqrt
from itertools import groupby
from typing import Tuple, Optional, Type, Union
from functools import partial

import torch
from torch import Tensor
from torch.types import _dtype
import torch.distributions as D
import warnings
from ot_vae_lightning.ot.matrix_utils import *

__all__ = [
    'w2_gaussian',
    'batch_w2_dissimilarity_gaussian_diag',
    'batch_w2_dissimilarity_gaussian',
    'batch_ot_gmm',
    'sinkhorn_log',
    'gaussian_barycenter',
    'compute_transport_operators',
    'apply_transport',
    'W2Mixin'
]

# ******************************************************************************************************************** #

def w2_gaussian(
        mean_source: Tensor,
        mean_target: Tensor,
        cov_source: Tensor,
        cov_target: Tensor,
        make_pd: bool = False,
        verbose: bool = False,
        dtype: Optional[_dtype] = torch.double
) -> Tensor:
    """
    Computes closed form squared W2 distance between Gaussian distribution_models (also known as Gelbrich Distance)
    :param mean_source: A 1-dim vectors representing the source distribution mean with optional leading batch dims [*, D]
    :param mean_target: A 1-dim vectors representing the target distribution mean with optional leading batch dims [*, D]
    :param cov_source: A 2-dim matrix representing the source distribution covariance [*, D, D]
    :param cov_target: A 2-dim matrix representing the target distribution covariance [*, D, D]
    :param make_pd: If ``True``, corrects matrices needed to be PD or PSD with by adding their minimum eigenvalue to
                    their diagonal.
    :param verbose: If ``True``, warns about correction value added to the diagonal of the matrices
    :param dtype: The type from which the result will be computed.

    :return: The squared Wasserstein 2 distance between N(mean_source, cov_source) and N(mean_target, cov_target)
    """
    mean_source, mean_target, cov_source, cov_target = _validate_args(  # noqa
        mean_source, 'mean_source', 'vec',
        mean_target, 'mean_target', 'vec',
        cov_source, 'cov_source', 'spd',
        cov_target, 'cov_target', 'spd',
        make_pd=make_pd, verbose=verbose, dtype=torch.double
    )

    cov_target_sqrt = sqrtm(cov_target)
    mix = cov_target_sqrt @ cov_source @ cov_target_sqrt

    mix, = _validate_args(
        mix, 'cov_target_sqrt @ cov_source @ cov_target_sqrt', 'spsd',
        make_pd=make_pd, verbose=verbose, dtype=dtype
    )

    mean_shift = torch.sum((mean_source - mean_target) ** 2, dim=-1)
    cov_shift_trace = torch.diagonal(cov_source + cov_target - 2 * sqrtm(mix), dim1=-2, dim2=-1).sum(dim=-1)
    return mean_shift + cov_shift_trace


# ******************************************************************************************************************** #


def batch_w2_dissimilarity_gaussian_diag(
        mean_source: Tensor,
        mean_target: Tensor,
        var_source: Tensor,
        var_target: Tensor,
        dtype: Optional[_dtype] = torch.double
) -> Tensor:
    r"""
    Computes the dissimilarity matrix:

    .. math::

        D_{bij} = W^{2}_{2}(\mathcal{N}(\mu_{bi}^{s} , \mathbf{I} \sigma_{bi}^{s}^{2}),
         \mathcal{N}(\mu_{bj}^{t} , \mathbf{I} \sigma_{bj}^{t}^{2}))

    :param mean_source: means of source distribution_models. [*, N, D]
    :param mean_target: means of target distribution_models. [*, M, D]
    :param var_source: vars of source distribution (scale). [*, N, D]
    :param var_target: vars of target distribution (scale). [*, M, D]
    :param dtype: The type from which the result will be computed.

    :return: Dissimilarity matrix D [*, N, M] where D[b, i, j] = W2(Sbi, Tbj).
    """
    mean_source, var_source = _validate_args(  # noqa
        mean_source, 'mean_source', 'vec',
        var_source, 'var_source', 'var',
        dtype=dtype

    )
    mean_target, var_target = _validate_args(  # noqa
        mean_target, 'mean_target', 'vec',
        var_target, 'var_target', 'var',
        dtype=dtype
    )

    dist_mean = (
            (mean_source**2).sum(-1, keepdim=True) +                                 # [*, N, 1]
            (mean_target**2).sum(-1).unsqueeze(-2) -                                 # [*, 1, M]
            2 * (mean_source @ mean_target.transpose(-2, -1))                        # [*, N, M]
    )

    dist_var = (
            var_source.sum(-1, keepdim=True) +                                       # [*, N, 1]
            var_target.sum(-1).unsqueeze(-2) -                                       # [*, 1, M]
            2 * (torch.sqrt(var_source) @ torch.sqrt(var_target.transpose(-2, -1)))  # [*, N, M]
    )

    w2 = dist_mean + dist_var
    return w2


# ******************************************************************************************************************** #


def batch_w2_dissimilarity_gaussian(
        mean_source: Tensor,
        mean_target: Tensor,
        cov_source: Tensor,
        cov_target: Tensor,
        make_pd: bool = False,
        verbose: bool = False,
        dtype: Optional[_dtype] = torch.double,
) -> Tensor:
    r"""
    Computes the dissimilarity matrix:

    .. math::

        D_{bij} = W^{2}_{2}(\mathcal{N}(\mu_{bi}^{s} , \Sigma_{bi}^{s}^{2}),
         \mathcal{N}(\mu_{bj}^{t} , \Sigma_{bj}^{t}^{2}))

    :param mean_source: means of source distribution_models. [*, N, D]
    :param mean_target: means of target distribution_models. [*, M, D]
    :param cov_source: covariance matrix of source distribution. [*, N, D, D]
    :param cov_target: covariance matrix of target distribution. [*, M, D, D]
    :param dtype: The type from which the result will be computed.
    :param make_pd: If ``True``, corrects matrices needed to be PD or PSD with by adding their minimum eigenvalue to
                    their diagonal.
    :param verbose: If ``True``, warns about correction value added to the diagonal of the matrices

    :return: Dissimilarity matrix D [*, N, M] where D[b, i, j] = W2(Sbi, Tbj).
    """
    mean_source, cov_source = _validate_args(  # noqa
        mean_source, 'mean_source', 'vec',
        cov_source, 'cov_source', 'spd',
        dtype=dtype
    )

    mean_target, cov_target = _validate_args(  # noqa
        mean_target, 'mean_target', 'vec',
        cov_target, 'cov_target', 'spd',
        dtype=dtype
    )

    N, M, D = mean_source.size(-2), mean_target.size(-2), mean_source.size(-1)

    ones = [1] * (mean_source.dim()-2)
    dissimilarity = w2_gaussian(
        mean_source.repeat_interleave(M, -2),           # [*, N*M, D]  [1,1,1,1,2,2,2,2,3...]
        mean_target.repeat(*ones, N, 1),                # [*, M*N, D]  [1,2,3,4,1,2,3,4,1...]
        cov_source.repeat_interleave(M, -3),            # [*, N*M, D, D]
        cov_target.repeat(*ones, N, 1, 1),              # [*, M*N, D, D]
        make_pd=make_pd, verbose=verbose, dtype=dtype
    )  # [*, N*M, N*M]  [11, 12, 13, 14, 21, 22, 23, 24,...]
    dissimilarity = dissimilarity.view(*mean_source.shape[:-2], N, M)
    return dissimilarity


# ******************************************************************************************************************** #


def batch_ot_gmm(
        mean_source: Tensor,
        mean_target: Tensor,
        cov_source: Tensor,
        cov_target: Tensor,
        diag: bool,
        weight_source: Optional[Tensor] = None,
        weight_target: Optional[Tensor] = None,
        verbose: bool = False,
        dtype: Optional[_dtype] = torch.double,
        **sinkhorn_kwargs
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes the entropy-regularized squared W2 distance[1] between the following gaussian mixtures:

    .. math::

        GMM_{s}=\sum_{i} w_{bi}^{s} \mathcal{N}(\mu_{bi}^{s} , \sigma_{bi}^{s}^{2})

        GMM_{t}=\sum_{i} w_{bi}^{t} \mathcal{N}(\mu_{bi}^{t} , \sigma_{bi}^{t}^{2})

    if weight_source or weight_target is None, equal probability for each component is assumed. Inspired from [2].

    [1] Marco Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transport
    [2] Yongxin Chen, Tryphon T. Georgiou and Allen Tannenbaum Optimal transport for Gaussian mixture models

    :param mean_source: batch of means of source distribution_models. [*, N, D]
    :param mean_target: batch of means of target distribution_models. [*, M, D]
    :param cov_source: covariance matrix of source distribution. [*, N, D, D] ([*, N, D] if `diag=True`)
    :param cov_target: covariance matrix of source distribution. [*, M, D, D] ([*, N, D] if `diag=True`)
    :param diag: If ``True`` expects variance vectors instead of covariance matrices
    :param weight_source: Probability vector of source GMM distribution. [*, N]
    :param weight_target: Probability vector of target GMM distribution. [*, M]
    :param verbose: If ``True``, warns about correction value added to the diagonal of the matrices
    :param dtype: The type from which the result will be computed.

    :return: Total W2 cost between GMMs and GMMt SUM_{i,j} cost(i, j) * pi(i, j)
             Coupling matrix pi [*, N, M]
    """
    weight_source = torch.ones_like(mean_source.select(dim=-1, index=0)) / mean_source.size(-2) \
        if weight_source is None else weight_source
    weight_target = torch.ones_like(mean_target.select(dim=-1, index=0)) / mean_target.size(-2) \
        if weight_target is None else weight_target

    mean_source, cov_source, weight_source = _validate_args(  # noqa
        mean_source, 'mean_source', 'vec',
        cov_source, 'cov_source', 'var' if diag else 'spd',
        weight_source, 'weight_source', 'prob',
        dtype=dtype
    )

    mean_target, cov_target, weight_target = _validate_args(  # noqa
        mean_target, 'mean_target', 'vec',
        cov_target, 'cov_target', 'var' if diag else 'spd',
        weight_target, 'weight_target', 'prob',
        dtype=dtype
    )

    if diag:
        cost_matrix = batch_w2_dissimilarity_gaussian_diag(
            mean_source, mean_target, cov_source, cov_target, dtype=dtype
        )
    else:
        cost_matrix = batch_w2_dissimilarity_gaussian(
            mean_source, mean_target, cov_source, cov_target,
            make_pd=True, verbose=verbose, dtype=dtype
        )  # TODO: This gives NaN !

    max_per_mat = cost_matrix.max(-2, keepdim=True)[0].max(-1, keepdim=True)[0]
    coupling = sinkhorn_log(weight_source, weight_target, cost_matrix/max_per_mat, **sinkhorn_kwargs)

    # elem-wise multiplication and sum => SUM cost(i,j) pi(i, j)
    total_cost = torch.sum(cost_matrix * coupling, dim=(-2, -1))
    return total_cost, coupling


# ******************************************************************************************************************** #


def sinkhorn_log(
        a: Tensor,
        b: Tensor,
        C: Tensor,
        reg: float = 1e-5,
        max_iter: int = 1000,
        threshold: float = STABILITY_CONST,
) -> Tensor:
    r"""
    W2 Optimal Transport under entropic regularisation using fixed point sinkhorn iteration [1]
    in log domain for increased numerical stability.
    Inspired from `Wohlert <http://gist.github.com/wohlert/8589045ab544082560cc5f8915cc90bd>`_'s
    implementation.

    [1] Marco Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transport

    :param a: probability vector. [*, N]
    :param b: probability vector. [*, M]
    :param C: cost matrix. [*, N, M]
    :param reg: entropic regularisation weight. Default = 1e-3
    :param max_iter: max number of fixed point iterations
    :param threshold: stopping threshold of total variation between successive iterations

    :return: Coupling matrix (optimal transport plan). [*, N, M]
    """
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)
    log_a = torch.log(a + STABILITY_CONST)
    log_b = torch.log(b + STABILITY_CONST)
    Cr = - C / reg

    for i in range(max_iter):
        u0, v0 = u, v

        # v^{l+1} = b / (K^T u^(l+1)), u^{l+1} = a / (K v^l)
        v = log_b - torch.logsumexp(Cr + u.unsqueeze(-1), dim=-2)
        u = log_a - torch.logsumexp(Cr + v.unsqueeze(-2), dim=-1)

        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        if torch.min(diff).item() < threshold: break

    # Transport plan pi = diag(a)*K*diag(b)
    pi = torch.exp(u.unsqueeze(-1) + v.unsqueeze(-2) + Cr)
    return pi


# ******************************************************************************************************************** #


def gaussian_barycenter(
        mean: Tensor,
        cov: Tensor,
        weights: Tensor,
        diag: bool,
        n_iter: int = 100,
        dtype: Optional[_dtype] = torch.double
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes the W2 Barycenter of the Gaussian distribution_models Normal(MU[i], diagC[i]) with weight weight_source[i]
    according to the fixed point method introduced in [1].

    .. math::

            \mu_{barycenter} = \sum_{i} w_{i} * \mu_{i}

            \sigma_{barycenter} = \sqrt{\sum_{i} (w_{i} * \sigma_{i})^{2}}

    [1] P. C. Alvarez-Esteban, E. del Barrio, J. Cuesta-Albertos, and C. Matran,
    A fixed-point approach to barycenters in Wasserstein space

    :param mean: mean components. [*, N, D]
    :param cov: covariance components. [*, N, D, D] ([*, N, D] if `diag==True`)
    :param weights: Batch of probability vector. [*, N]
    :param diag: If ``True`` expects variance vectors instead of covariance matrices
    :param n_iter: number of fixed points iterations when `diag == False`
    :param dtype: The type from which the result will be computed.

    :return: :math: `\mu_{barycenter}, \sigma_{barycenter}^2` [*, D], [*, D, D] (or [*, D] if `diag==True`)
    """
    mean, cov, weights = _validate_args(  # noqa
        mean, 'mean', 'vec',
        cov, 'cov', 'var' if diag else 'spd',
        weights, 'weights', 'prob',
        dtype=dtype
    )

    # aggregate the means
    weights = weights.unsqueeze(-2)                            # [*, N]    (unsqueeze) --> [*, 1, N]
    mean_b = (weights @ mean)                                  # [*, 1, N] x [*, N, D] --> [*, 1, D]
    mean_b = mean_b.squeeze(-2)                                # [*, 1, D]   (squeeze) --> [*, D]

    # compute the covariance
    if diag:
        cov_b = (weights @ torch.sqrt(cov)) ** 2               # [*, 1, N] x [*, N, D] --> [*, 1, D]
        cov_b = cov_b.squeeze(-2)                              # [*, 1, D]   (squeeze) --> [*, D]
        return mean_b, cov_b

    weights = weights.squeeze(-2).unsqueeze(-1).unsqueeze(-1)  # [*, 1, N] --> [*, N, 1, 1]

    # init with randomly drawn cov
    random_idx = torch.randint(size=(1,), high=cov.size(-3)).item()
    cov_b = cov.select(dim=-3, index=random_idx)               # [*, N, D, D] (select) --> [*, D, D]
    cov_b = cov_b.unsqueeze(-3)                                # [*, D, D] (unsqueeze) --> [*, 1, D, D]
    for _ in range(n_iter):
        sqrt_cov_b = sqrtm(cov_b)
        mix = sqrt_cov_b @ cov @ sqrt_cov_b                    # [*, 1, D, D] x  [*, N, D, D] x [*, 1, D, D]
        cov_b = (weights * sqrtm(mix)).sum(-3, keepdim=True)   # [*, 1, D, D] x. [*, N, D, D] (sum) --> [*, 1, D, D]
    cov_b = cov_b.squeeze(-3)

    return mean_b, cov_b


# ******************************************************************************************************************** #


def compute_transport_operators(
        cov_source: Tensor,
        cov_target: Tensor,
        stochastic: bool,
        diag: bool,
        pg_star: float = 0,
        make_pd: bool = False,
        verbose: bool = False,
        dtype: Optional[_dtype] = torch.double
) -> Tuple[Tensor, Tensor]:
    r"""
    Batch implementation of eq. 17, 19 in [1]

    .. math::

        (17) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{-0.5}_{s} (\Sigma^{0.5}_{s} \Sigma_{t}
         \Sigma^{0.5}_{s})^{0.5} \Sigma^{-0.5}_{s} + P/G^{*} I

        (19) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{+0.5}_{t} (\Sigma^{0.5}_{t} \Sigma_{s}
         \Sigma^{0.5}_{t})^{0.5} \Sigma^{-0.5}_{t} \Sigma^{\dagger}_{s} + P/G^{*} I

             \Sigma_w = \Sigma^{0.5}_{t} (I - \Sigma^{0.5}_{t} T^{*} \Sigma^{\dagger}_{s} T^{*}
              \Sigma^{0.5}_{t}) \Sigma^{0.5}_{t},  T^{*} = T_{t \longrightarrow s} (17)

    [1] D. Freirich, T. Michaeli and R. Meir.
    `A Theory of the Distortion-Perception Tradeoff in Wasserstein Space <https://proceedings.neurips.cc/paper/2021/
    hash/d77e68596c15c53c2a33ad143739902d-Abstract.html>`_

    :param cov_source: Batch of SPD matrices. Source covariances. [*, D1, D1] (or [*, D1] if ``diag==True``)
    :param cov_target: Batch of SPSD matrices. Target covariances. [*, D2, D2] (or [*, D2] if ``diag==True``)
    :param stochastic: If ``True`` return (T_{s -> t}, \Sigma_w) of (19) else return (T_{s -> t}, `None`) (17)
    :param diag: If ``True`` expects cov_source and cov_target to be batch of vectors (representing diagonal matrices)
    :param pg_star: Perception-distortion ratio. can be seen as temperature.
     (`p_gstar=0` --> best perception, `p_gstar=1` --> best distortion). Default `0`.
    :param make_pd: If ``True``, corrects matrices needed to be PD or PSD with by adding their minimum eigenvalue to
                their diagonal.
    :param verbose: If ``True``, warns about correction value added to the diagonal of the matrices
    :param dtype: The type from which the result will be computed.

    :return: Batch of transport operators T_{s -> t} and \Sigma_w
    """
    if stochastic and diag: cov_source[cov_source < STABILITY_CONST] = 0
    cov_source, cov_target = _validate_args(  # noqa
        cov_source, 'cov_source', 'var' if diag else ('spsd' if stochastic else 'spd'),
        cov_target, 'cov_target', 'var' if diag else ('spd' if stochastic else 'spsd'),
        make_pd=make_pd, dtype=dtype, verbose=verbose
    )

    stochastic_warn = """
    The noise covariance matrix is not positive definite. 
    Falling back to the non-stochastic implementation
    """

    if diag and stochastic:
        T, Cw = _compute_transport_diag_stochastic(cov_source, cov_target, pg_star)
        if (Cw <= 0).any() and verbose: warnings.warn(stochastic_warn); stochastic = False

    if diag and not stochastic:
        T, Cw = _compute_transport_diag(cov_source, cov_target, pg_star)

    if not diag and stochastic:
        T, Cw = _compute_transport_full_mat_stochastic(cov_source, cov_target, pg_star)
        if not is_spd(Cw, strict=True).all() and verbose: warnings.warn(stochastic_warn); stochastic = False

    if not diag and not stochastic:
        T, Cw = _compute_transport_full_mat(cov_source, cov_target, pg_star)

    return T, Cw  # noqa


# ******************************************************************************************************************** #


def apply_transport(
        input: Tensor,
        mean_source: Tensor,
        mean_target: Tensor,
        T: Tensor,
        Cw: Optional[Tensor] = None,
        diag: bool = False,
        make_pd: bool = False,
        verbose: bool = False,
        dtype: Optional[_dtype] = torch.double
) -> Tensor:
    r"""
    Executes optimal W2 transport of the sample given as `input` using the provided mean and transport operators (T, Cw)
    in batch fashion according to eq. 17 and eq. 19 in [1]

    .. math::

        (17) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{-0.5}_{s} (\Sigma^{0.5}_{s} \Sigma_{t}
         \Sigma^{0.5}_{s})^{0.5} \Sigma^{-0.5}_{s} + P/G^{*} I

        (19) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{+0.5}_{t} (\Sigma^{0.5}_{t} \Sigma_{s}
         \Sigma^{0.5}_{t})^{0.5} \Sigma^{-0.5}_{t} \Sigma^{\dagger}_{s} + P/G^{*} I

             \Sigma_w = \Sigma^{0.5}_{t} (I - \Sigma^{0.5}_{t} T^{*} \Sigma^{\dagger}_{s} T^{*}
              \Sigma^{0.5}_{t}) \Sigma^{0.5}_{t},  T^{*} = T_{t \longrightarrow s} (17)

    [1] D. Freirich, T. Michaeli and R. Meir.
    `A Theory of the Distortion-Perception Tradeoff in Wasserstein Space <https://proceedings.neurips.cc/paper/2021/
    hash/d77e68596c15c53c2a33ad143739902d-Abstract.html>`_

    :param input: Batch of samples from the source distribution, to transport to the target distribution. [*, D1]
    :param mean_source: Mean of the source distribution. [*, D1]
    :param mean_target: Mean of the target distribution. [*, D2]
    :param T: transport Operator from the source to the target distribution. [*, D2, D1] (or [*, D1] if ``diag==True``)
    :param Cw: Noise covariance if the source distribution is degenerate. [*, D2, D1] (or [*, D1] if ``diag==True``]
    :param diag: If ``True`` expects T and Cw to be vectors (representing diagonal matrices)
    :param make_pd: If ``True``, corrects matrices needed to be PD or PSD with by adding their minimum eigenvalue to
                    their diagonal.
    :param verbose: If ``True``, warns about correction value added to the diagonal of the matrices
    :param dtype: The type from which the result will be computed.

    :return: T (input - mean_source) + mean_target + W,   W~Normal(0, Cw). [*, D2]
    """
    ignore_cw = Cw is None or torch.allclose(Cw, torch.zeros_like(Cw))
    input, mean_source, mean_target, T, Cw = _validate_args(  # noqa
        input, 'input', 'vec',
        mean_source, 'mean_source', 'vec',
        mean_target, 'mean_target', 'vec',
        T, 'T', 'vec' if diag else 'mat',
        Cw, 'Cw', ('vec' if diag else 'mat') if ignore_cw else ('var' if diag else 'spd'),
        make_pd=make_pd, verbose=verbose, dtype=dtype
    )  # TODO: will raise false error if D1 != D2

    x_centered = input - mean_source
    x_transported = T * x_centered if diag else (T @ x_centered.unsqueeze(-1)).squeeze(-1)

    x_transported = x_transported + mean_target

    if not ignore_cw:
        mu = torch.zeros_like(mean_target)
        noise = D.Normal(mu, Cw) if diag else D.MultivariateNormal(mu, Cw)
        x_transported += noise.sample()

    return x_transported


# ******************************************************************************************************************** #


class W2Mixin(object):
    def __init__(self, **kwargs):
        self._orig_kwargs = kwargs
        self.stochastic = kwargs.pop('stochastic', False)
        self.diag = kwargs.pop('diag', False)
        self.pg_star = kwargs.pop('pg_star', 0.)
        self.make_pd = kwargs.pop('make_pd', False)
        self.verbose = kwargs.pop('verbose', False)
        self.dtype = kwargs.pop('dtype', torch.double)

        self.mean_cov = partial(mean_cov, diag=self.diag)
        self.batch_w2_dissimilarity_gaussian_diag = partial(batch_w2_dissimilarity_gaussian_diag, dtype=self.dtype)
        self.batch_w2_dissimilarity_gaussian = partial(batch_w2_dissimilarity_gaussian, make_pd=self.make_pd, verbose=self.verbose, dtype=self.dtype)
        self.batch_ot_gmm = partial(batch_ot_gmm, diag=self.diag, verbose=self.verbose, dtype=self.dtype)
        self.gaussian_barycenter = partial(gaussian_barycenter, diag=self.diag, dtype=self.dtype)
        self.compute_transport_operators = partial(
            compute_transport_operators, diag=self.diag, stochastic=self.stochastic, pg_star=self.pg_star,
            make_pd=self.make_pd, verbose=self.verbose, dtype=self.dtype
        )

    def get_var_normal(self, distribution: Union[D.Normal, D.MultivariateNormal]):
        return distribution.variance if self.diag else distribution.covariance_matrix

    def instantiate_normal(self, *args, **kwargs):
        if self.diag:
            kwargs.pop('covariance_matrix', None)
            kwargs.pop('precision_matrix', None)
            kwargs.pop('scale_tril', None)
            return D.Independent(D.Normal(*args, **kwargs), 1)
        else:
            kwargs.pop('scale', None)
            return D.MultivariateNormal(*args, **kwargs)

    def w2_gaussian(
            self,
            mean_source: Tensor,
            mean_target: Tensor,
            cov_source: Tensor,
            cov_target: Tensor,
    ) -> Tensor:
        return w2_gaussian(
            mean_source,
            mean_target,
            torch.diag_embed(cov_source) if self.diag else cov_source,
            torch.diag_embed(cov_target) if self.diag else cov_target,
            make_pd=self.make_pd, verbose=self.verbose, dtype=self.dtype
        )

    def apply_transport(
            self,
            inputs: Tensor,
            mean_source: Tensor,
            mean_target: Tensor,
            T: Tensor,
            Cw: Tensor,
            batch_dim: Optional[int] = None,
    ) -> Tensor:
        return apply_transport(
            inputs,
            mean_source.unsqueeze(batch_dim) if batch_dim is not None else mean_source,
            mean_target.unsqueeze(batch_dim) if batch_dim is not None else mean_target,
            T.unsqueeze(batch_dim - bool(not self.diag)) if batch_dim is not None else T,
            Cw.unsqueeze(batch_dim - bool(not self.diag)) if batch_dim is not None else Cw,
            diag=self.diag, make_pd=self.make_pd, verbose=self.verbose, dtype=self.dtype
        )

    def __repr__(self):
        return ', '.join([f'{k}={v}' for k,v in self._orig_kwargs.items()])

# ******************************************************************************************************************** #


def _validate_args(
        *args,
        make_pd: bool = False,
        verbose: bool = False,
        dtype: Optional[_dtype] = None,
        tol: float = 1e-5
) -> Tuple[Tensor]:
    dim_stack = []
    comp_stack = []
    batch_dims = []
    new_args = []
    has_components = 'prob' in args
    for arg, name, arg_type in zip(args[::3], args[1::3], args[2::3]):
        assert isinstance(name, str), "Flow assertion Error: name should be a string"
        if not isinstance(arg, Tensor):
            raise ValueError(f"""
            `{name}` is expected to be a torch.Tensor, got `{type(arg)}` instead.
            """)
        if arg_type in ['vec', 'var']:
            if arg.dim() < 1 + int(has_components):
                raise ValueError(f"""
                `{name}` should be 1-dim vectors {'with a leading component dimension' if has_components else ''}
                (+ optional leading batch dimensions), got `{name}.dim()={arg.dim()}`
                """)
            if arg_type == 'var':
                if (arg < 0).any():
                    raise ValueError(f"""
                    `{name}` is expected to be a valid variance vector with positive entries.
                    """)
            dim_stack.append(arg.size(-1))
            if has_components:
                comp_stack.append(arg.size(-2))
                batch_dims.append(arg.shape[:-2])
            else:
                batch_dims.append(arg.shape[:-1])
            new_args.append(arg.to(dtype) if dtype is not None else arg)
        elif arg_type == 'prob':
            if arg.dim() < 1:
                raise ValueError(f"""
                `{name}` should be a 1-dim vectors (+ optional leading batch dimensions),
                 got `{name}.dim()={arg.dim()}`
                """)
            if (arg < -tol).any() or (arg.sum(-1) < 1-tol).any() or (arg.sum(-1) > 1+tol).any():
                raise ValueError(f"""
                `{name}` is expected to be a valid probability vector with positive entries that sum up to 1.
                """)
            comp_stack.append(arg.size(-1))
            batch_dims.append(arg.shape[:-1])
            new_args.append(arg.to(dtype) if dtype is not None else arg)
        elif arg_type in ['mat', 'spd', 'spsd', 'pd', 'psd']:
            if arg.dim() < 2 + int(has_components):
                raise ValueError(f"""
                `{name}` should be a 2-dim matrix {'with a leading component dimension' if has_components else ''}
                 (+ optional leading batch dimensions), got `{name}.dim()`={arg.dim()}.
                 """)

            if arg_type[0] == 's' and not is_symmetric(arg).all():
                raise ValueError(f"""
                `{name}` should be symmetric. Found {(~is_symmetric(arg)).int().sum().item()} non-symmetric matrices.
                """)
            strict = 's' not in (arg_type[1:] if arg_type[0] == 's' else arg_type)
            semi = '' if strict else 'semi '
            if 'pd' in arg_type and not is_pd(arg, strict=strict).all():
                if make_pd:
                    arg, val = make_psd(arg, strict=strict, return_correction=True)
                    if verbose:
                        warnings.warn(f"""
                        `{name}` is not positive {semi}definite. Adding a small value to the 
                        diagonal (<{val.max().item():.2e}) to ensure the matrices are positive {semi}definite
                        """)
                else:
                    raise ValueError(f"""
                    `{name}` should be symmetric and positive {semi}definite. Use `make_pd=True` to automatically add
                     a small value to the matrix diagonals.
                    """)
            new_args.append(arg.to(dtype) if dtype is not None else arg)
            dim_stack.extend([arg.size(-2), arg.size(-1)])
            if 'prob' in args:
                comp_stack.append(arg.size(-3))
                batch_dims.append(arg.shape[:-3])
            else:
                batch_dims.append(arg.shape[:-2])

        def all_equal(iterable):
            if len(dim_stack) < 2: return True
            g = groupby(iterable)
            return next(g, True) and not next(g, False)

        if not all_equal(dim_stack):
            raise ValueError(f"""
            All the inputs dimensionalities should match,
            got {dim_stack}
            """)
        if not all_equal(batch_dims) and torch.broadcast_shapes(*batch_dims) not in batch_dims:  # allow for broadcast
            raise ValueError(f"""
            All the inputs leading batch dimensions should be broadcastable,
            got {batch_dims}
            """)
        if not all_equal(comp_stack):
            raise ValueError(f"""
            All the inputs component dimension should match,
            got {batch_dims}
            """)
    return tuple(new_args)


# ******************************************************************************************************************** #


def _compute_transport_diag(
        cov_source: Tensor,
        cov_target: Tensor,
        p_gstar
) -> Tuple[Tensor, Tensor]:
    """
    Helper function for `compute_transport_operators`.
    This function doesn't have any parameter checking and was not designed to be standalone.
    Use with care.
    """
    T = (1 - p_gstar) * torch.sqrt(cov_target / cov_source + STABILITY_CONST) + p_gstar
    return T, torch.zeros_like(T)


# ******************************************************************************************************************** #


def _compute_transport_diag_stochastic(
        cov_source: Tensor,
        cov_target: Tensor,
        p_gstar: float
) -> Tuple[Tensor, Tensor]:
    """
    Helper function for `compute_transport_operators`.
    This function doesn't have any parameter checking and was not designed to be standalone.
    Use with care.
    """
    T_star = torch.sqrt(cov_source / cov_target + STABILITY_CONST)

    # Compute SVD pseudo-inverse of `cov_source`
    pinv_source = cov_source.clone()
    pinv_source[pinv_source > STABILITY_CONST] = 1 / pinv_source[pinv_source > STABILITY_CONST]
    pinv_source[pinv_source <= STABILITY_CONST] = 0

    T = (1 - p_gstar) * torch.sqrt(cov_target * cov_source) * pinv_source + p_gstar
    var_w = sqrt(1 - p_gstar) * cov_target * (1 - cov_target * pinv_source * T_star ** 2)
    return T, var_w


# ******************************************************************************************************************** #


def _compute_transport_full_mat(
        cov_source: Tensor,
        cov_target: Tensor,
        p_gstar: float
) -> Tuple[Tensor, Tensor]:
    """
    Helper function for `compute_transport_operators`.
    This function doesn't have any parameter checking and was not designed to be standalone.
    Use with care.
    """
    sqrtCs, isqrtCs = sqrtm(cov_source), invsqrtm(cov_source + STABILITY_CONST * eye_like(cov_source))
    T = (1 - p_gstar) * (isqrtCs @ sqrtm(sqrtCs @ cov_target @ sqrtCs) @ isqrtCs) + p_gstar * eye_like(cov_source)
    return T, torch.zeros_like(T)


# ******************************************************************************************************************** #


def _compute_transport_full_mat_stochastic(
        cov_source: Tensor,
        cov_target: Tensor,
        pg_star: float
) -> Tuple[Tensor, Tensor]:
    """
    Helper function for `compute_transport_operators`.
    This function doesn't have any parameter checking and was not designed to be standalone.
    Use with care.
    """
    Identity = eye_like(cov_source)
    pinv_source = torch.linalg.pinv(cov_source)
    sqrt_target, inv_sqrt_target = sqrtm(cov_target), invsqrtm(cov_target + STABILITY_CONST * Identity)

    # Roles are swapped on purpose, cov_source might only be positive semi definite
    T_star = _compute_transport_full_mat(cov_source=cov_target, cov_target=cov_source, p_gstar=0)[0]

    T = (1 - pg_star) * (sqrt_target @ sqrtm(sqrt_target @ cov_source @ sqrt_target) @ inv_sqrt_target @ pinv_source) + pg_star * Identity
    Cw = sqrt(1 - pg_star) * sqrt_target @ (Identity - sqrt_target @ T_star @ pinv_source @ T_star @ sqrt_target) @ sqrt_target
    return T, Cw

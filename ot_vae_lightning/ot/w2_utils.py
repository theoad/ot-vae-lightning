"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of Wasserstein 2 Optimal transport utilities

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from math import sqrt
import torch
from torch import Tensor
from typing import Tuple, Optional
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal

from ot_vae_lightning.ot.matrix_utils import *

__all__ = [
    'compute_transport_operators',
    'apply_transport',
    'w2_gaussian',
    'batch_w2_dissimilarity_gaussian_diag',
    'batch_w2_gmm_diag',
    'sinkhorn_log',
    'gaussian_barycenter_diag'
]


# ******************************************************************************************************************** #


def compute_transport_operators(
        cov_source: Tensor,
        cov_target: Tensor,
        stochastic: bool,
        diag: bool,
        pg_star: float = 0,
) -> Tuple[Tensor, Optional[Tensor]]:
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

    :param cov_source: Batch of SPD matrices. Source covariances. [B, D] or [B, D, D] (also supports broadcast [D])
    :param cov_target: Batch of SPSD matrices. Target covariances. [B, D] or [B, D, D] (also supports broadcast [D])
    :param stochastic: If ``True`` return (T_{s -> t}, \Sigma_w) of (19) else return (T_{s -> t}, `None`) (17)
    :param diag: If ``True`` expects cov_source and cov_target to be batch of vectors (representing diagonal matrices)
    :param pg_star: Perception-distortion ratio. can be seen as temperature.
     (`p_gstar=0` --> best perception, `p_gstar=1` --> best distortion). Default `0`.

    :return: Batch of transport operators T_{s -> t} and \Sigma_w
    """
    # TL;DR
    # Sorry for the spaghetti function. Error checking is better this way than with modular calls.
    # Parameter validation is done in this wrapper function and actual computations are done in modular calls.

    # ----------------------------------------------- PARAMETER CHECKS ----------------------------------------------- #
    if not 0 <= pg_star <= 1:
        raise ValueError(f"pg_star must be in the interval [0, 1], got {pg_star}")

    if cov_source.isinf().any():
        raise ValueError(f"Found {cov_source.isinf().nonzero().size(0)} `inf` elements in `cov_source`")
    if cov_source.isnan().any():
        raise ValueError(f"Found {cov_source.isnan().nonzero().size(0)} `nan` elements in `cov_source`")
    if cov_target.isinf().any():
        raise ValueError(f"Found {cov_source.isinf().nonzero().size(0)} `inf` elements in `cov_target`")
    if cov_target.isnan().any():
        raise ValueError(f"Found {cov_source.isnan().nonzero().size(0)} `nan` elements in `cov_target`")

    if diag:
        # cov_source and cov_target are assumed to be batch of vectors representing diagonal matrices
        if cov_source.dim() not in [1, 2] or cov_target.dim() not in [1, 2]:
            raise ValueError(f"""
            `diag`=True: `cov_source` and `cov_target` should be 1-dim or 2-dim tensors (batch of diagonals),
             got {cov_source.dim()} and {cov_target.dim()}
            """)

        # explicit broadcast
        squeeze = cov_source.dim() == 1 and cov_target.dim() == 1
        if cov_source.dim() == 1:
            cov_source = cov_source.unsqueeze(0)
        if cov_target.dim() == 1:
            cov_target = cov_target.unsqueeze(0)

        if cov_source.size(1) != cov_target.size(1):
            raise ValueError(f"""
            `cov_source` and `cov_target` should have the same dimensionality,
             got {cov_source.size(1)} and {cov_target.size(1)}
            """)

        if stochastic:
            if (cov_source < 0).any():
                raise ValueError(f"""
                `diag`=True, `stochastic`=True: In this configuration, `cov_source` should be non-negative (as diagonals
                 of positive semi-definite matrices). Found {(cov_source < 0).count_nonzero().item()} negative elements.
                """)
            if (cov_target <= 0).any():
                raise ValueError(f"""
                `diag`=True, `stochastic`=True: In this configuration, `cov_target` should be positive (as diagonals of
                 positive definite matrices). Found {(cov_target <= 0).count_nonzero().item()} non-positive elements.
                """)
        else:
            if (cov_source <= 0).any():
                raise ValueError(f"""
                `diag`=True, `stochastic`=False: In this configuration, `cov_source` should be positive (as diagonal of
                 positive definite matrices). Found {(cov_source <= 0).count_nonzero().item()} non-positive elements.
                """)
            if (cov_target < 0).any():
                raise ValueError(f"""
                `diag`=True, `stochastic`=False: In this configuration, `cov_source` should be non-negative (as diagonals
                 of positive semi-definite matrices). Found {(cov_target < 0).count_nonzero().item()} negative elements.
                """)
    else:
        if cov_source.dim() not in [2, 3] or cov_target.dim() not in [2, 3]:
            raise ValueError(f"""
            `cov_source` and `cov_target` should be 2-dim or 3-dim tensors (batch of matrices),
             got {cov_source.dim()} and {cov_target.dim()}
            """)

        # explicit broadcast
        squeeze = cov_source.dim() == 2 and cov_target.dim() == 2
        if cov_source.dim() == 2:
            cov_source = cov_source.unsqueeze(0)
        if cov_target.dim() == 2:
            cov_target = cov_target.unsqueeze(0)

        if cov_source.shape[1:] != cov_target.shape[1:]:
            raise ValueError(f"""
            `cov_source` and cov_target should have the same dimensionality,
             got {cov_source.shape[1:]} and {cov_target.shape[1:]}
            """)

        if stochastic:
            if not is_spsd(cov_source).all():
                raise ValueError(f"""
                `diag`=False, `stochastic`=True: In this configuration, `cov_source` should be a batch of symmetric and
                positive semi-definite matrices. Found {(torch.logical_not(is_spsd(cov_source))).count_nonzero().item()}
                matrices in the batch that are not SPSD.
                """)
            if not is_spd(cov_target).all():
                raise ValueError(f"""
                `diag`=False, `stochastic`=True: In this configuration, `cov_target` should be a batch of symmetric and
                positive definite matrices. Found {(torch.logical_not(is_spd(cov_target))).count_nonzero().item()}
                matrices in the batch that are not SPD.
                """)
        else:
            if not is_spd(cov_source).all():
                raise ValueError(f"""
                `diag`=False, `stochastic`=False: In this configuration, `cov_source` should be a batch of symmetric and
                positive definite matrices. Found {(torch.logical_not(is_spd(cov_source))).count_nonzero().item()}
                matrices in the batch that are not SPD.
                """)
            if not is_spsd(cov_target).all():
                raise ValueError(f"""
                `diag`=False, `stochastic`=False: In this configuration, `cov_target` should be a batch of symmetric and
                positive semi-definite matrices. Found {(torch.logical_not(is_spsd(cov_target))).count_nonzero().item()}
                matrices in the batch that are not SPSD.
                """)
    # ------------------------------------------- END OF PARAMETER CHECKS -------------------------------------------- #

    if diag:
        if stochastic:
            T, Cw = _compute_transport_diag_stochastic(cov_source, cov_target, pg_star)
        else:
            T, Cw = _compute_transport_diag(cov_source, cov_target, pg_star)
    else:
        if stochastic:
            T, Cw = _compute_transport_full_mat_stochastic(cov_source, cov_target, pg_star)
        else:
            T, Cw = _compute_transport_full_mat(cov_source, cov_target, pg_star)

    T = T.squeeze(0) if squeeze else T
    Cw = Cw.squeeze(0) if squeeze and Cw is not None else Cw
    return T, Cw


# ******************************************************************************************************************** #


def apply_transport(
        input: Tensor,
        mean_source: Tensor,
        mean_target: Tensor,
        T: Tensor,
        Cw: Optional[Tensor] = None,
        diag: bool = False,
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

    :param input: Batch of samples from the source distribution, to transport to the target distribution. [N, D1]
    :param mean_source: Mean of the source distribution. [D1]
    :param mean_target: Mean of the target distribution. [D2]
    :param T: transport Operator from the source to the target distribution. [D2, D1]
    :param Cw: Noise covariance if the source distribution is degenerate. [D2, D2]
    :param diag: If ``True`` expects T and Cw vectors (representing diagonal matrices)

    :return: T (input - mean_source) + mean_target + W,   W~Normal(0, Cw). [N, D2]
    """
    # ----------------------------------------------- PARAMETER CHECKS ----------------------------------------------- #
    if input.dim() != 2:
        raise ValueError(f"`input` should be a 2-dim matrix (batch of vectors), got input.dim()={input.dim()}")
    if mean_source.dim() != 1:
        raise ValueError(f"`mean_source` should be a 1-dim vector, got mean_source.dim()={mean_source.dim()}")
    if mean_target.dim() != 1:
        raise ValueError(f"`mean_target` should be a 1-dim vector, got mean_target.dim()={mean_target.dim()}")
    if diag and T.dim() != 1:
        raise ValueError(f"`diag`=True: `T` should be a 1-dim vector, got T.dim()={T.dim()}")
    if not diag and T.dim() != 2:
        raise ValueError(f"`diag`=False: `T` should be a 2-dim matrix, got T.dim()={T.dim()}")
    if mean_source.size(0) != input.size(1):
        raise ValueError(f"`mean_source` should have the same dimensionality as `input`, got {mean_source.size(0)}"
                         f" and {input.size(1)}")
    if T.size(-1) != input.size(1):
        raise ValueError(f"`T` should have the same dimensionality as `input`, got {T.size(-1)} and {input.size(1)}")
    if T.size(0) != mean_target.size(0):
        raise ValueError(f"`mean_target` should have the same dimensionality as `T`'s first dimension, got {T.size(0)}"
                         f" and {mean_target.size(0)}")
    if Cw is not None:
        if diag and Cw.dim() != 1:
            raise ValueError(f"`diag`=True: `Cw` should be a 1-dim vector, got Cw.dim()={Cw.dim()}")
        if not diag and Cw.dim() != 2:
            raise ValueError(f"`diag`=False: `Cw` should be a 2-dim matrix, got Cw.dim()={Cw.dim()}")
        if Cw.size(0) != T.size(0):
            raise ValueError(f"`Cw` should have the same dimensionality as `T`'s first dimension,"
                             f" got {T.size(0)} and {Cw.size(0)}")
        if Cw.dim() == 2 and not is_spd(Cw):
            raise ValueError(f"As matrix, `Cw` should be a symmetric and positive definite")
        if Cw.dim() == 1 and (Cw <= 0).any():
            raise ValueError(f"As vector, `Cw` should contain only positive values")
    # ------------------------------------------- END OF PARAMETER CHECKS -------------------------------------------- #

    w = 0
    if Cw is not None:
        w = torch.zeros_like(Cw.size(0))
        if diag:
            w = Normal(w, Cw).rsample()
        else:
            w = MultivariateNormal(w, Cw).rsample()

    x_centered = input - mean_source
    if diag:
        x_transported = T * x_centered
    else:
        x_transported = torch.matmul(T, x_centered.T).T

    x_transported = x_transported + mean_target + w
    return x_transported


# ******************************************************************************************************************** #


def w2_gaussian(
        mean_source: Tensor,
        mean_target: Tensor,
        cov_source: Tensor,
        cov_target: Tensor
) -> Tensor:
    """
    Computes closed form squared W2 distance between Gaussians
    :param mean_source: A 1-dim vectors representing the source distribution mean
    :param mean_target: A 1-dim vectors representing the target distribution mean
    :param cov_source: A 2-dim matrix representing the source distribution covariance
    :param cov_target: A 2-dim matrix representing the target distribution covariance
    :return: The squared Wasserstein 2 distance between N(mean_source, cov_source) and N(mean_target, cov_target)
    """
    if mean_source.dim() != 1:
        raise ValueError(f"`mean_source` should be a 1-dim vectors representing the source distribution mean,"
                         f" got mean_source.dim()={mean_source.dim()}")
    if mean_target.dim() != 1:
        raise ValueError(f"`mean_target` should be a 1-dim vectors representing the target distribution mean,"
                         f" got mean_source.dim()={mean_target.dim()}")
    if cov_source.dim() != 2:
        raise ValueError(f"`cov_source` should be a 2-dim matrix representing the source distribution covariance,"
                         f" got cov_source.dim()={cov_source.dim()}")
    if cov_target.dim() != 2:
        raise ValueError(f"`cov_target` should be a 2-dim matrix representing the target distribution covariance,"
                         f" got cov_target.dim()={cov_target.dim()}")
    if not (mean_source.size(0) == mean_source.size(0) == cov_source.size(0) == cov_source.size(1) == cov_target.size(0) == cov_target.size(1)):
        raise ValueError(f"All the inputs dimensions should match,"
                         f" got {mean_source.size(0)}, {mean_source.size(0)}, {cov_source.size(0)}, {cov_source.size(1)}"
                         f", {cov_target.size(0)}, {cov_target.size(1)}")
    if not is_spsd(cov_source):
        raise ValueError("`cov_source` should be symmetric and positive semi-definite")
    if not is_spsd(cov_target):
        raise ValueError("`cov_target` should be symmetric and positive semi-definite")
    cov_target_sqrt = sqrtm(cov_target)
    squared_w2 = (
            torch.linalg.vector_norm(mean_source - mean_target) ** 2 +
            torch.trace(cov_source + cov_target - 2 * sqrtm(cov_target_sqrt @ cov_source @ cov_target_sqrt))
    )
    return squared_w2


# ******************************************************************************************************************** #


def batch_w2_dissimilarity_gaussian_diag(
        mean_source: Tensor,
        mean_target: Tensor,
        var_source: Tensor,
        var_target: Tensor
) -> Tensor:
    r"""
    Computes the dissimilarity matrix:

    .. math::

        D_{bij} = W^{2}_{2}(\mathcal{N}(\mu_{bi}^{s} , \sigma_{bi}^{s}^{2}),
         \mathcal{N}(\mu_{bj}^{t} , \sigma_{bj}^{t}^{2}))

    :param mean_source: batch of means of source distributions. [B, N, D...]
    :param mean_target: batch of means of target distributions. [B, M, D...]
    :param var_source: batch of vars of source distribution (scale). [B, N, D...]
    :param var_target: batch of vars of target distribution (scale). [B, M, D...]

    :return: Dissimilarity matrix D [B, N, M] where D[b, i, j] = W2(Sbi, Tbj).
    """
    if not (mean_source.size() == mean_target.size() == var_source.size() == var_target.size()):
        raise ValueError(f"""
        All the input parameters are expected to have the same shape,
        got {mean_source.size()}, {mean_target.size()}, {var_source.size()} and {var_target.size()} 
        which are not compatible with the function implementation.
        """)
    if mean_source.dim() < 3 or mean_target.dim() < 3 or var_source.dim() < 3 or var_target.dim():
        raise ValueError(f"""
        All the input parameters are expected to have at least 3 dimensions,
        got {mean_source.dim()}, {mean_target.dim()}, {var_source.dim()} and {var_target.dim()} 
        which are not compatible with the function implementation. Please unsqueeze the Tensor to obtain
        batch of sequences of tensors.
        """)
    mean_source = mean_source.flatten(2)
    mean_target = mean_target.flatten(2)
    var_source = var_source.flatten(2)
    var_target = var_target.flatten(2)

    dist_mean = (
            (mean_source**2).sum(-1, keepdim=True) +                                 # [B, N, 1]
            (mean_target**2).sum(-1).unsqueeze(-2) -                                 # [B, 1, M]
            2 * (mean_source @ mean_target.transpose(-2, -1))                        # [B, N, M]
    )

    dist_var = (
            var_source.sum(-1, keepdim=True) +                                       # [B, N, 1]
            var_target.sum(-1).unsqueeze(-2) -                                       # [B, 1, M]
            2 * (torch.sqrt(var_source) @ torch.sqrt(var_target.transpose(-2, -1)))  # [B, N, M]
    )

    w2 = dist_mean + dist_var
    return w2


# ******************************************************************************************************************** #


def batch_w2_gmm_diag(
        mean_source: Tensor,
        mean_target: Tensor,
        var_source: Tensor,
        var_target: Tensor,
        weight_source: Optional[Tensor] = None,
        weight_target: Optional[Tensor] = None,
        **kwargs
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes the entropy-regularized squared W2 distance[1] between the following gaussian mixtures:

    .. math::

        GMM_{s}=\sum_{i} w_{bi}^{s} \mathcal{N}(\mu_{bi}^{s} , \sigma_{bi}^{s}^{2})

        GMM_{t}=\sum_{i} w_{bi}^{t} \mathcal{N}(\mu_{bi}^{t} , \sigma_{bi}^{t}^{2})

    if weight_source or weight_target is None, equal probability for each component is assumed. Inspired from [2].

    [1] Marco Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transport
    [2] Yongxin Chen, Tryphon T. Georgiou and Allen Tannenbaum Optimal transport for Gaussian mixture models

    :param mean_source: batch of means of source distributions. [B, N, D...]
    :param mean_target: batch of means of target distributions. [B, M, D...]
    :param var_source: batch of vars of source distribution (scale). [B, N, D...]
    :param var_target: batch of vars of target distribution (scale). [B, M, D...]
    :param weight_source: Probability vector of source GMM distribution. [B, N...]
    :param weight_target: Probability vector of target GMM distribution. [B, M...]

    :return: Total W2 cost between GMMs and GMMt SUM_{i,j} cost(i, j) * pi(i, j)
             Coupling matrix pi [B, N, M]
    """
    # TODO: parameter checking
    weight_source = torch.ones_like(mean_source[0, :, 0])/mean_source.size(1) if weight_source is None else weight_source
    weight_target = torch.ones_like(mean_target[0, :, 0])/mean_target.size(1) if weight_target is None else weight_target
    cost_matrix = batch_w2_dissimilarity_gaussian_diag(mean_source, mean_target, var_source, var_target)
    coupling = sinkhorn_log(weight_source, weight_target, cost_matrix/cost_matrix.max(), **kwargs)

    # elem-wise multiplication and sum => SUM cost(i,j) pi(i, j)
    total_cost = torch.sum(cost_matrix * coupling, dim=(-2, -1))
    return total_cost, coupling


# ******************************************************************************************************************** #


def sinkhorn_log(
        a: Tensor,
        b: Tensor,
        C: Tensor,
        reg: float = 1e-5,
        max_iter: int = 10,
        threshold: float = 1e-2,
) -> Tensor:
    r"""
    W2 Optimal Transport under entropic regularisation using fixed point sinkhorn iteration [1]
    in log domain for increased numerical stability.
    Inspired from `Wohlert <httweight_source://gist.github.com/wohlert/8589045ab544082560cc5f8915cc90bd>`_'s
    implementation.

    [1] Marco Cuturi, Sinkhorn Distances: Lightspeed Computation of Optimal Transport

    :param a: probability vector. [B, N]
    :param b: probability vector. [B, M]
    :param C: cost matrix. [B, N, M]
    :param reg: entropic regularisation weight. Default = 1e-3
    :param max_iter: max number of fixed point iterations
    :param threshold: stopping threshold of total variation between successive iterations

    :return: Coupling matrix (optimal transport plan). [B, N, M]
    """
    # TODO: parameter checking
    def log_boltzmann_kernel(K: Tensor, u: Tensor, v: Tensor):
        return (u.unsqueeze(-1) + v.unsqueeze(-2) - K) / reg

    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    for i in range(max_iter):
        u0, v0 = u, v

        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(C, u, v)
        u_ = torch.log(a + STABILITY_CONST) - torch.logsumexp(K, dim=-1)
        u = reg * u_ + u

        # v^{l+1} = b / (K^T u^(l+1))
        K = log_boltzmann_kernel(C, u, v).transpose(-2, -1)
        v_ = torch.log(b + STABILITY_CONST) - torch.logsumexp(K, dim=-1)
        v = reg * v_ + v

        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        if torch.min(diff).item() < threshold:
            break

    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(C, u, v)
    pi = torch.exp(K)
    return pi


# ******************************************************************************************************************** #


def gaussian_barycenter_diag(
        mean: Tensor,
        var: Tensor,
        weights: Tensor
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes the W2 Barycenter of the Gaussian distributions Normal(MU[i], diagC[i]) with weight weight_source[i]
    according to the fixed point method introduced in [1].

    .. math::

            \mu_{barycenter} = \sum_{i} w_{i} * \mu_{i}

            \sigma_{barycenter} = \sqrt{\sum_{i} (w_{i} * \sigma_{i})^{2}}

    [1] P. C. Alvarez-Esteban, E. del Barrio, J. Cuesta-Albertos, and C. Matran,
    A fixed-point approach to barycenters in Wasserstein space

    :param mean: mean components. [C, D]
    :param var: vars components. [C, D]
    :param weights: Batch of probability vector. [N, C]

    :return: :math: `\mu_{barycenter}, \sigma_{barycenter}^2`
    """
    # TODO: parameter checking
    mean_b = torch.matmul(weights, mean)
    var_b = torch.matmul(weights, torch.sqrt(var)) ** 2
    return mean_b, var_b


# ******************************************************************************************************************** #


def _compute_transport_diag(
        cov_source: Tensor,
        cov_target: Tensor,
        p_gstar: float
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Helper function for `compute_transport_operators`.
    This function doesn't have any parameter checking and was not designed to be standalone.
    Use with care.
    """
    return (1 - p_gstar) * torch.sqrt(cov_target / cov_source + STABILITY_CONST) + p_gstar, None


# ******************************************************************************************************************** #


def _compute_transport_diag_stochastic(
        cov_source: Tensor,
        cov_target: Tensor,
        p_gstar: float
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Helper function for `compute_transport_operators`.
    This function doesn't have any parameter checking and was not designed to be standalone.
    Use with care.
    """
    T_star = torch.sqrt(cov_source / cov_target + STABILITY_CONST)

    # Compute SVD pseudo-inverse of `cov_source`
    pinv_source = cov_source.clone()
    pinv_source[pinv_source > STABILITY_CONST] = 1 / pinv_source[pinv_source > STABILITY_CONST]

    T = (1 - p_gstar) * torch.sqrt(cov_target * cov_source) * pinv_source + p_gstar
    var_w = sqrt(1 - p_gstar) * cov_target * (1 - cov_target * pinv_source * T_star ** 2)
    return T, var_w

# ******************************************************************************************************************** #


def _compute_transport_full_mat(
        cov_source: Tensor,
        cov_target: Tensor,
        p_gstar: float
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Helper function for `compute_transport_operators`.
    This function doesn't have any parameter checking and was not designed to be standalone.
    Use with care.
    """
    Identity = torch.eye(*cov_source.shape[1:2], out=torch.empty_like(cov_source))
    sqrtCs, isqrtCs = sqrtm(cov_source), invsqrtm(cov_source + STABILITY_CONST * Identity)
    T = (1 - p_gstar) * (isqrtCs @ sqrtm(sqrtCs @ cov_target @ sqrtCs) @ isqrtCs) + p_gstar * Identity
    return T, None


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
    Identity = torch.eye(*cov_source.shape[1:2], out=torch.empty_like(cov_source))
    pinv_source = torch.linalg.pinv(cov_source)
    sqrt_target, inv_sqrt_target = sqrtm(cov_target), invsqrtm(cov_target + STABILITY_CONST * Identity)

    # Roles are swapped on purpose, cov_source might only be positive semi definite
    T_star = _compute_transport_full_mat(cov_source=cov_target, cov_target=cov_source, p_gstar=0)

    T = (1-pg_star) * (sqrt_target @ sqrtm(sqrt_target @ cov_source @ sqrt_target) @ inv_sqrt_target @ pinv_source) + pg_star * Identity
    Cw = sqrt(1-pg_star) * sqrt_target @ (Identity - sqrt_target @ T_star @ pinv_source @ T_star @ sqrt_target) @ sqrt_target
    return T, Cw

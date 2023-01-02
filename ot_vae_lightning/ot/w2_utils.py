"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of Wasserstein 2 utilities

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from itertools import groupby
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.types import _dtype
import warnings
from ot_vae_lightning.ot.matrix_utils import *
from pytorch_lightning.utilities import rank_zero_info

__all__ = [
    'w2_gaussian',
    'batch_w2_dissimilarity_gaussian_diag',
    'batch_w2_dissimilarity_gaussian',
    'batch_ot_gmm',
    'sinkhorn_log',
    'gaussian_barycenter'
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
    Computes closed form squared W2 distance between Gaussian distributions (also known as Gelbrich Distance)
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
    mean_source, mean_target, cov_source, cov_target = validate_args(  # noqa
        mean_source, 'mean_source', 'vec',
        mean_target, 'mean_target', 'vec',
        cov_source, 'cov_source', 'spd',
        cov_target, 'cov_target', 'spd',
        make_pd=make_pd, verbose=verbose, dtype=torch.double
    )

    cov_target_sqrt = sqrtm(cov_target)
    mix = cov_target_sqrt @ cov_source @ cov_target_sqrt

    mix, = validate_args(
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

    :param mean_source: means of source distributions. [*, N, D]
    :param mean_target: means of target distributions. [*, M, D]
    :param var_source: vars of source distribution (scale). [*, N, D]
    :param var_target: vars of target distribution (scale). [*, M, D]
    :param dtype: The type from which the result will be computed.

    :return: Dissimilarity matrix D [*, N, M] where D[b, i, j] = W2(Sbi, Tbj).
    """
    mean_source, var_source = validate_args(  # noqa
        mean_source, 'mean_source', 'vec',
        var_source, 'var_source', 'var',
        dtype=dtype

    )
    mean_target, var_target = validate_args(  # noqa
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

    :param mean_source: means of source distributions. [*, N, D]
    :param mean_target: means of target distributions. [*, M, D]
    :param cov_source: covariance matrix of source distribution. [*, N, D, D]
    :param cov_target: covariance matrix of target distribution. [*, M, D, D]
    :param dtype: The type from which the result will be computed.
    :param make_pd: If ``True``, corrects matrices needed to be PD or PSD with by adding their minimum eigenvalue to
                    their diagonal.
    :param verbose: If ``True``, warns about correction value added to the diagonal of the matrices

    :return: Dissimilarity matrix D [*, N, M] where D[b, i, j] = W2(Sbi, Tbj).
    """
    mean_source, cov_source = validate_args(  # noqa
        mean_source, 'mean_source', 'vec',
        cov_source, 'cov_source', 'spd',
        dtype=dtype
    )

    mean_target, cov_target = validate_args(  # noqa
        mean_target, 'mean_target', 'vec',
        cov_target, 'cov_target', 'spd',
        dtype=dtype
    )

    N, M, D = mean_source.size(-2), mean_target.size(-2), mean_source.size(-1)
    if verbose: rank_zero_info(f"got {N} source components and {M} target components")

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

    :param mean_source: batch of means of source distributions. [*, N, D]
    :param mean_target: batch of means of target distributions. [*, M, D]
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

    mean_source, cov_source, weight_source = validate_args(  # noqa
        mean_source, 'mean_source', 'vec',
        cov_source, 'cov_source', 'var' if diag else 'spd',
        weight_source, 'weight_source', 'prob',
        dtype=dtype
    )

    mean_target, cov_target, weight_target = validate_args(  # noqa
        mean_target, 'mean_target', 'vec',
        cov_target, 'cov_target', 'var' if diag else 'spd',
        weight_target, 'weight_target', 'prob',
        dtype=dtype
    )

    if verbose: rank_zero_info("computing cost matrix...")

    if diag:
        cost_matrix = batch_w2_dissimilarity_gaussian_diag(
            mean_source, mean_target, cov_source, cov_target, dtype=dtype
        )
    else:
        cost_matrix = batch_w2_dissimilarity_gaussian(
            mean_source, mean_target, cov_source, cov_target,
            make_pd=True, verbose=verbose, dtype=dtype
        )

    if verbose: rank_zero_info("cost matrix computed:", cost_matrix)

    max_per_mat = cost_matrix.max(-2, keepdim=True)[0].max(-1, keepdim=True)[0]
    coupling = sinkhorn_log(
        weight_source, weight_target, cost_matrix/max_per_mat, **sinkhorn_kwargs
    )

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
    Computes the W2 Barycenter of the Gaussian distributions Normal(MU[i], diagC[i]) with weight weight_source[i]
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
    mean, cov, weights = validate_args(  # noqa
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


class DimError(Exception):
    pass

def validate_args(
        *args,
        make_pd: bool = False,
        verbose: bool = False,
        dtype: Optional[_dtype] = None,
        tol: float = STABILITY_CONST
) -> Tuple[Tensor]:
    dim_stack = []
    comp_stack = []
    batch_dims = []
    new_args = []
    has_components = 'prob' in args
    for arg, name, arg_type in zip(args[::3], args[1::3], args[2::3]):
        assert isinstance(name, str), "Flow assertion Error: name should be a string"
        if not isinstance(arg, Tensor):
            raise DimError(f"""
            `{name}` is expected to be a torch.Tensor, got `{type(arg)}` instead.
            """)
        if arg_type in ['vec', 'var']:
            if arg.dim() < 1 + int(has_components):
                raise DimError(f"""
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
                raise DimError(f"""
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
        elif arg_type in ['spd', 'spsd', 'pd', 'psd']:
            if arg.dim() < 2 + int(has_components):
                raise DimError(f"""
                `{name}` should be a 2-dim matrix {'with a leading component dimension' if has_components else ''}
                 (+ optional leading batch dimensions), got `{name}.dim()`={arg.dim()}.
                 """)

            if arg_type[0] == 's' and not is_symmetric(arg).all():
                raise ValueError(f"""
                `{name}` should be symmetric. Found {(~is_symmetric(arg)).int().sum().item()} non-symmetric matrices.
                """)
            strict = 's' not in (arg_type[1:] if arg_type[0] == 's' else arg_type)
            semi = '' if strict else 'semi '
            if not is_pd(arg, strict=strict).all():
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
            raise DimError(f"""
            All the inputs dimensionalities should match,
            got {dim_stack}
            """)
        if not all_equal(batch_dims) and torch.broadcast_shapes(*batch_dims) not in batch_dims:  # allow for broadcast
            raise DimError(f"""
            All the inputs leading batch dimensions should be broadcastable,
            got {batch_dims}
            """)
        if not all_equal(comp_stack):
            raise DimError(f"""
            All the inputs component dimension should match,
            got {batch_dims}
            """)
    return tuple(new_args)

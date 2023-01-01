"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of Wasserstein 2 Optimal transport utilities

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Tuple, Optional
from math import sqrt

import torch
from torch import Tensor, logical_not
from torch.types import _dtype
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
import warnings

from ot_vae_lightning.ot.matrix_utils import *

__all__ = [
    'compute_transport_operators',
    'apply_transport'
]


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

    :param cov_source: Batch of SPD matrices. Source covariances. [B, D] or [B, D, D] (also supports broadcast [D])
    :param cov_target: Batch of SPSD matrices. Target covariances. [B, D] or [B, D, D] (also supports broadcast [D])
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
    # TL;DR
    # @theoad: Sorry for the spaghetti function. Error checking is better this way than with modular calls.
    # Parameter validation is done in this wrapper function and actual computations are done in modular calls.

    # ----------------------------------------- PARAMETER CHECKS + BROADCAST ----------------------------------------- #
    if not 0 <= pg_star <= 1:
        raise ValueError(f"pg_star must be in the interval [0, 1], got {pg_star}")

    if cov_source.isinf().any():
        raise ValueError(f"Found {cov_source.isinf().nonzero().size(0)} `inf` elements in `cov_source`")
    if cov_source.isnan().any():
        raise ValueError(f"Found {cov_source.isnan().nonzero().size(0)} `nan` elements in `cov_source`")
    if cov_target.isinf().any():
        raise ValueError(f"Found {cov_target.isinf().nonzero().size(0)} `inf` elements in `cov_target`")
    if cov_target.isnan().any():
        raise ValueError(f"Found {cov_target.isnan().nonzero().size(0)} `nan` elements in `cov_target`")

    if dtype is not None:
        if cov_source.dtype != dtype:
            cov_source = cov_source.to(dtype=dtype)
        if cov_target.dtype != dtype:
            cov_target = cov_target.to(dtype=dtype)

    if diag:
        # cov_source and cov_target are assumed to be batch of vectors representing diagonal matrices
        if cov_source.dim() not in [1, 2] or cov_target.dim() not in [1, 2]:
            raise ValueError(f"""
            `diag`=True: `cov_source` and `cov_target` should be 1-dim or 2-dim tensors (batch of diagonals),
             got cov_source.dim()={cov_source.dim()} and cov_target.dim()={cov_target.dim()}.
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
             got cov_source.size(1)={cov_source.size(1)} and cov_target.size(1)={cov_target.size(1)}.
            """)

        cov_source_neg = cov_source < 0 if stochastic else cov_source <= 0
        cov_target_neg = cov_target <= 0 if stochastic else cov_target < 0
        if cov_source_neg.any():
            semi = 'semi-' if stochastic else ''
            if make_pd:
                val = cov_source.clamp(min=0 if stochastic else STABILITY_CONST) - cov_source
                cov_source = cov_source.clamp(min=0 if stochastic else STABILITY_CONST)
                if verbose:
                    warnings.warn(f"""
                    `cov_source` is not positive {semi}definite. Adding a small value to the 
                    diagonal (<{'{:.2e}'.format(val.max().item())}) to ensure the matrices are positive {semi}definite.
                    """)
            raise ValueError(f"""
            `diag`=True, `stochastic`={stochastic}: In this configuration, `cov_source` should be
             {'non-negative' if stochastic else 'positive'} (as diagonals of positive {semi}definite matrices).
             Found {cov_source_neg.count_nonzero().item()}/{cov_source.numel()}
             {'negative' if stochastic else 'non-positive'} elements in `cov_source`.
            """)
        if stochastic: cov_source[cov_source < STABILITY_CONST] = 0

        if cov_target_neg.any():
            semi = '' if stochastic else 'semi-'
            if make_pd:
                val = cov_target.clamp(min=0 if stochastic else STABILITY_CONST) - cov_target
                cov_target = cov_target.clamp(min=0 if stochastic else STABILITY_CONST)
                if verbose:
                    warnings.warn(f"""
                    `cov_target` is not positive {semi}definite. Adding a small value to the 
                    diagonal (<{'{:.2e}'.format(val.max().item())}) to ensure the matrices are positive {semi}definite.
                    """)
            raise ValueError(f"""
            `diag`=True, `stochastic`={stochastic}: In this configuration, `cov_target` should be
             {'positive' if stochastic else 'non-negative'} (as diagonals of positive {semi}definite matrices).
             Found {cov_target_neg.count_nonzero().item()}/{cov_target.numel()}
             {'non-positive' if stochastic else 'negative'} elements in `cov_target`.
            """)
    else:  # not diag
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
            `cov_source` and `cov_target` should have the same dimensionality,
             got {cov_source.shape[1:]} and {cov_target.shape[1:]}
            """)

        cov_source_spd = is_spd(cov_source, strict=not stochastic)
        cov_target_spd = is_spd(cov_target, strict=stochastic)

        if not cov_source_spd.all():
            semi = 'semi-' if stochastic else ''
            if is_symmetric(cov_source).all() and make_pd:
                cov_source, val = make_psd(cov_source, strict=not stochastic, return_correction=True)
                if verbose:
                    warnings.warn(f"""
                    `cov_source` is not positive {semi}definite. Adding a small value to the 
                    diagonal (<{'{:.2e}'.format(val.max().item())}) to ensure the matrices are positive {semi}definite.
                    """)
            else:
                num_not_psd = logical_not(cov_source_spd).count_nonzero().item()
                num_psd = cov_source_spd.count_nonzero().item()
                raise ValueError(f"""
                `diag`=False, `stochastic`={stochastic}: In this configuration, `cov_source` should be a batch of
                 symmetric and positive {semi}definite matrices.
                 Found {num_not_psd}/{num_psd + num_not_psd} matrices in the batch that are not positive {semi}definite.
                 In order to automatically add small value to the matrices diagonal use `make_pd`=True.
                """)

        if not cov_target_spd.all():
            semi = '' if stochastic else 'semi-'
            if is_symmetric(cov_target).all() and make_pd:
                cov_target, val = make_psd(cov_target, strict=stochastic, return_correction=True)
                if verbose:
                    warnings.warn(f"""
                    `cov_target` is not positive {semi}definite. Adding a small value to the 
                    diagonal (<{'{:.2e}'.format(val.max().item())}) to ensure the matrices are positive {semi}definite.
                    """)
            else:
                num_not_psd = logical_not(cov_target_spd).count_nonzero().item()
                num_psd = cov_target_spd.count_nonzero().item()
                raise ValueError(f"""
                `diag`=False, `stochastic`={stochastic}: In this configuration, `cov_target` should be a batch of
                 symmetric and positive {semi}definite matrices.
                 Found {num_not_psd}/{num_psd + num_not_psd} matrices in the batch that are not positive {semi}definite.
                 In order to add small value to the matrices diagonal use `make_pd`=True.
                """)
    # ------------------------------------------- END OF PARAMETER CHECKS -------------------------------------------- #

    if diag:
        if stochastic:
            T, Cw = _compute_transport_diag_stochastic(cov_source, cov_target, pg_star)
            if (Cw <= 0).any():
                warnings.warn(f"""
                The noise covariance matrix is not positive definite. Falling back to the non-stochastic implementation
                """)
                T, Cw = _compute_transport_diag(cov_source, cov_target, pg_star)
        else:
            T, Cw = _compute_transport_diag(cov_source, cov_target, pg_star)
    else:
        if stochastic:
            T, Cw = _compute_transport_full_mat_stochastic(cov_source, cov_target, pg_star)
            if not is_spd(Cw, strict=True).all():
                warnings.warn(f"""
                The noise covariance matrix is not positive definite. Falling back to the non-stochastic implementation
                """)
                T, Cw = _compute_transport_full_mat(cov_source, cov_target, pg_star)
        else:
            T, Cw = _compute_transport_full_mat(cov_source, cov_target, pg_star)

    T = T.squeeze(0) if squeeze else T
    Cw = Cw.squeeze(0) if squeeze else Cw
    return T, Cw


# ******************************************************************************************************************** #


def apply_transport(
        input: Tensor,
        mean_source: Tensor,
        mean_target: Tensor,
        T: Tensor,
        Cw: Optional[Tensor] = None,
        diag: bool = False,
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

    :param input: Batch of samples from the source distribution, to transport to the target distribution. [B, D1]
    :param mean_source: Mean of the source distribution. [D1]
    :param mean_target: Mean of the target distribution. [D2]
    :param T: transport Operator from the source to the target distribution. [D2, D1]
    :param Cw: Noise covariance if the source distribution is degenerate. [D2, D2]
    :param diag: If ``True`` expects T and Cw to be vectors (representing diagonal matrices)
    :param dtype: The type from which the result will be computed.

    :return: T (input - mean_source) + mean_target + W,   W~Normal(0, Cw). [N, D2]
    """
    # ----------------------------------------------- PARAMETER CHECKS ----------------------------------------------- #
    if dtype is not None:
        if input.dtype != dtype: input = input.to(dtype=dtype)
        if mean_source.dtype != dtype: mean_source = mean_source.to(dtype=dtype)
        if mean_target.dtype != dtype: mean_target = mean_target.to(dtype=dtype)
        if T.dtype != dtype: T = T.to(dtype=dtype)
        if Cw.dtype != dtype: Cw = mean_source.to(dtype=dtype)

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
        raise ValueError(f""" `mean_source` should have the same dimensionality as `input`, 
        got {mean_source.size(0)} and {input.size(1)}""")
    if T.size(-1) != input.size(1):
        raise ValueError(f"`T` should have the same dimensionality as `input`, got {T.size(-1)} and {input.size(1)}")
    if T.size(0) != mean_target.size(0):
        raise ValueError(f"""`mean_target` should have the same dimensionality as `T`'s first dimension, 
        got {T.size(0)} and {mean_target.size(0)}""")
    if Cw is not None and not torch.allclose(Cw, torch.zeros_like(Cw)):
        if diag and Cw.dim() != 1:
            raise ValueError(f"`diag`=True: `Cw` should be a 1-dim vector, got Cw.dim()={Cw.dim()}")
        if not diag and Cw.dim() != 2:
            raise ValueError(f"`diag`=False: `Cw` should be a 2-dim matrix, got Cw.dim()={Cw.dim()}")
        if Cw.size(0) != T.size(0):
            raise ValueError(f"""`Cw` should have the same dimensionality as `T`'s first dimension, 
            got {T.size(0)} and {Cw.size(0)}""")
        if Cw.dim() == 2 and not is_spd(Cw, strict=True):
            raise ValueError(f"As matrix, `Cw` should be a symmetric and positive definite")
        if Cw.dim() == 1 and (Cw <= 0).any():
            raise ValueError(f"As vector, `Cw` should contain only positive values")
    # ------------------------------------------- END OF PARAMETER CHECKS -------------------------------------------- #

    w = torch.zeros_like(mean_target)
    if Cw is not None and not torch.allclose(Cw, torch.zeros_like(Cw)):
        if diag:
            w = Normal(w, Cw).rsample()
        else:
            w = MultivariateNormal(w, Cw).rsample()

    x_centered = input - mean_source
    x_transported = T * x_centered if diag else torch.matmul(T, x_centered.T).T

    x_transported = x_transported + mean_target + w
    return x_transported


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

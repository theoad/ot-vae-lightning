from typing import Union, Tuple

import torch
from torch import Tensor, BoolTensor

__all__ = [
    'eye_like',
    'sqrtm',
    'logm',
    'expm',
    'invsqrtm',
    'is_spd',
    'is_pd',
    'is_symmetric',
    'min_eig',
    'make_psd',
    'STABILITY_CONST'
]

STABILITY_CONST = 1e-8


# Function from the pyRiemann package ported in pytorch
def _matrix_operator(matrices: Tensor, operator) -> Tensor:
    """
    Matrix equivalent of an operator. Works batch-wise
    Porting of pyRiemann to pyTorch
    Original Author: Alexandre Barachant
    https://github.com/alexandrebarachant/pyRiemann
    """
    eigvals, eigvects = torch.linalg.eigh(matrices, UPLO='L')
    eigvals = torch.diag_embed(operator(eigvals))
    return eigvects @ eigvals @ eigvects.transpose(-2, -1)


def eye_like(matrices: Tensor) -> Tensor:
    """
    Return Identity matrix with the same shape, device and dtype as matrices

    :param matrices: Batch of matrices with shape [*, C, D] where * is zero or leading batch dimensions
    :return: Tensor T with shape [*, C, D]. with T[i] = torch.eye(C, D)
    """
    return torch.eye(*matrices.shape[-2:-1], out=torch.empty_like(matrices)).expand_as(matrices)


def sqrtm(matrices: Tensor) -> Tensor:
    """
    :param matrices: batch of SPSD matrices
    :returns: batch containing mat. square root of each matrix
    """
    return _matrix_operator(matrices, torch.sqrt)


def logm(matrices: Tensor) -> Tensor:
    """
    :param matrices: batch of SPD matrices
    :returns: batch containing mat. log of each matrix
    """
    return _matrix_operator(matrices, torch.log)


def expm(matrices: Tensor) -> Tensor:
    """"
    :param matrices: batch of SPSD matrices
    :returns: batch containing mat. exp of each matrix
    """
    return _matrix_operator(matrices, torch.exp)


def invsqrtm(matrices: Tensor) -> Tensor:
    """
    :param matrices: batch of SPD matrices
    :returns: batch containing mat. inverse sqrt. of each matrix
    """
    isqrt = lambda x: 1. / torch.sqrt(x)
    return _matrix_operator(matrices, isqrt)


def is_symmetric(matrices: Tensor) -> BoolTensor:
    """
    Boolean method. Checks if matrix is symmetric.

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :return: Boolean tensor T with shape [*]. with T[i] == True <=> matrices[i] is symmetric
    """
    if matrices.size(-1) != matrices.size(-2):
        return torch.full_like(matrices.mean(dim=(-1, -2)), 0).bool()  # = Tensor([False, False, ..., False])
    return torch.sum((matrices - matrices.transpose(-2, -1))**2, dim=(-1, -2)) < STABILITY_CONST


def min_eig(matrices: Tensor) -> Tensor:
    """
    Returns the minimal eigen values of a batch of matrices (signed).

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :return: Tensor T with shape [*]. with T[i] = min(eig(matrices[i])
    """
    return torch.min(torch.linalg.eigh(matrices)[0], dim=-1)[0]


def is_pd(matrices: Tensor, strict=True) -> BoolTensor:
    """
    Boolean method. Checks if matrices are Positive Definite (PD).

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :param strict: If ``False`` checks the matrices are positive semi-definite
    :return: Boolean tensor T with shape [*]. with T[i] == True <=> matrices[i] is PD
    """
    return min_eig(matrices) > 0 if strict else min_eig(matrices) >= 0


def is_spd(matrices: Tensor, strict=True) -> BoolTensor:
    """
    Boolean method. Checks if matrices are Symmetric and Positive Definite (SPD).

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :param strict: If ``False`` checks the matrices are positive semi-definite (SPSD)
    :return: Boolean tensor T with shape [*]. with T[i] == True <=> matrices[i] is SPD
    """
    return torch.logical_and(is_symmetric(matrices), is_pd(matrices, strict=strict)).bool()


def make_psd(matrices: Tensor, strict: bool = False, return_correction: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Add to each matrix its minimal eigen value to make it positive definite.

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :param strict: If ``True``, add a small stability constant to make the matrices positive definite (PD)
    :param return_correction: If ``True``, returns the correction added to the diagonal of the matrices.
    :return: Tensor T with shape [*]. with T[i] = matrices[i] + min(eig(matrices[i]) * I
    """
    small_val = torch.abs(min_eig(matrices).clamp(max=0))
    if strict:
        small_val += STABILITY_CONST
    res = matrices + small_val * eye_like(matrices)
    if return_correction:
        return res, small_val
    return res

import torch
from torch import Tensor

__all__ = ['sqrtm', 'logm', 'expm', 'invsqrtm', 'is_spd', 'is_spsd', 'STABILITY_CONST']

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
    eigvals = torch.diag_embed(operator(_around(eigvals)))
    return _around(eigvects) @ _around(eigvals) @ _around(eigvects).transpose(-2, -1)


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


def is_spd(matrices: Tensor) -> Tensor:
    """
    Boolean method. Checks if matrix is Symmetric Positive Definite (SPD).

    :param matrices: Batch of matrices with shape [N, D, D]. Also supports [D, D] (regular matrix)
    :return: Boolean tensor T with shape [N]. with T[i] == True <=> matrices[i] is SPD
    """
    if matrices.dim() == 2:
        return torch.logical_and(
            torch.sum((matrices - matrices.T)**2) < STABILITY_CONST,
            torch.linalg.eigh(matrices)[0][0] > 0
        )
    return torch.logical_and(
            torch.sum((matrices - matrices.transpose(2, 1))**2, dim=(1, 2)) < STABILITY_CONST,  # symmetry
            torch.linalg.eigh(matrices)[0][:, 0] > 0                                            # positive definite
    )


def is_spsd(matrices: Tensor) -> Tensor:
    """
    Boolean method. Checks if matrix is Symmetric Positive Semi-Definite (SPSD).

    :param matrices: Batch of matrices of shape torch.Size([N, D, D]). Also supports [D, D] (regular matrix)
    :return: Boolean tensor T of shape torch.Size([N]). with T[i] == True <=> matrices[i] is SPSD
    """
    if matrices.dim() == 2:
        return torch.logical_and(
            torch.sum((matrices - matrices.T)**2) < STABILITY_CONST,
            torch.linalg.eigh(matrices)[0][0] >= 0
        )
    return torch.logical_and(
            torch.sum((matrices - matrices.transpose(2, 1))**2, dim=(1, 2)) < STABILITY_CONST,  # symmetry
            torch.linalg.eigh(matrices)[0][:, 0] >= 0                                           # positive semi-definite
    )


def _around(t: Tensor, decimals: float = 9) -> Tensor:
    """
    Rounding method to avoid numerical errors, torch equivalent of np.around(array, decimals=9)
    """
    return torch.round((t * 10 ** decimals))/10 ** decimals

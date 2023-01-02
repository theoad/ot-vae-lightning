"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a CI to validate W2 utilities vs. other, non-batched implementations

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
import torch
import numpy as np
import scipy.linalg as spl
from ot_vae_lightning.ot.w2_utils import *
from ot_vae_lightning.ot.matrix_utils import eye_like, STABILITY_CONST
from retrying import retry
import ot.bregman

_MAX_ATTEMPTS = 2
_DIM = 3

def _rand_mean_cov(leading_dims, dim):
    if isinstance(leading_dims, int): leading_dims = (leading_dims,)

    mean = torch.randn(*leading_dims, dim)
    cov = torch.randn(*leading_dims, dim, dim)
    # ensure matrix is SPD
    cov = cov @ cov.transpose(-1, -2) + eye_like(cov) * 1e-5
    return mean, cov

@retry(stop_max_attempt_number=_MAX_ATTEMPTS)
def test_w2_gaussian_same_yeilds_0(verbose=False):
    mean, cov = _rand_mean_cov((2,3), _DIM)
    res = w2_gaussian(mean, mean, cov, cov, make_pd=False, verbose=verbose)

    assert res.shape == torch.Size([2,3])
    assert torch.allclose(res, torch.zeros_like(res), atol=STABILITY_CONST * _DIM)
    if verbose: print(f'{"`w2_gaussian`":50s} {"0-test":20s} {"success":20s}')

@retry(stop_max_attempt_number=_MAX_ATTEMPTS)
def test_batch_w2_same_yeilds_0(verbose=False):
    mean, cov = _rand_mean_cov((2,3), _DIM)
    var = torch.diagonal(cov, dim1=-1, dim2=-2)  # extract variance
    assert var.shape == mean.shape == torch.Size([2,3,_DIM])

    # the penultimate dim is treated as component dim
    # [2,3,10] dist [2,3,10] --> [2,3,3]
    res = batch_w2_dissimilarity_gaussian_diag(mean, mean, var, var)

    assert res.shape == torch.Size([2,3,3])
    self_dissimilarity = torch.diagonal(res, dim1=-1, dim2=-2)
    assert self_dissimilarity.shape == torch.Size([2,3])
    assert torch.allclose(self_dissimilarity, torch.zeros_like(self_dissimilarity))
    if verbose: print(f'{"`batch_w2_dissimilarity_gaussian`":50s} {"0-test":20s} {"success":20s}')

@retry(stop_max_attempt_number=_MAX_ATTEMPTS)
def test_ot_gmm_same_yeilds_0(verbose=False):
    # distance between batch of 2 similar GMMs (with 3 comps) --> torch.tensor([0., 0.])
    mean, cov = _rand_mean_cov((2, 3), _DIM)
    var = torch.diagonal(cov, dim1=-1, dim2=-2)  # extract variance
    assert var.shape == mean.shape == torch.Size([2,3,_DIM])

    weights = torch.ones(2, 3)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # diagonal case
    res, _ = batch_ot_gmm(mean, mean, var, var, True, weights, None)  # None should be replace with uniform

    assert res.shape == torch.Size([2])
    assert torch.allclose(res, torch.zeros_like(res))

    # full matrix case
    res, _ = batch_ot_gmm(mean, mean, cov, cov, False, weights, None)  # None should be replace with uniform
    assert res.shape == torch.Size([2])
    assert torch.allclose(res, torch.zeros_like(res))

    if verbose: print(f'{"`batch_w2_gmm_diag`":50s} {"0-test":20s} {"success":20s}')

@retry(stop_max_attempt_number=_MAX_ATTEMPTS)
def test_gaussian_barycenter_same(verbose=False):
    mean, cov = _rand_mean_cov((2, 1), _DIM)

    # a batch of repeating mean and cov
    mean = mean.repeat(1,3,1)
    cov = cov.repeat(1,3,1,1)

    var = torch.diagonal(cov, dim1=-1, dim2=-2)  # extract variance
    assert var.shape == mean.shape == torch.Size([2, 3, _DIM])

    weights = torch.randn(2, 3, dtype=torch.double).abs()
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # diagonal case
    mean_b, var_b = gaussian_barycenter(mean, var, weights, diag=True)
    assert mean_b.shape == var_b.shape == torch.Size([2, _DIM])
    # make sure the mean of repeating vector is the equal to the vectors
    assert torch.allclose(mean_b, mean[:, 0].double())
    assert torch.allclose(var_b, var[:, 0].double())

    # full matrix case
    mean_b, cov_b = gaussian_barycenter(mean, cov, weights, diag=False)
    assert mean_b.shape == torch.Size([2, _DIM])
    assert cov_b.shape == torch.Size([2, _DIM, _DIM])
    assert torch.allclose(mean_b, mean[:, 0].double())
    assert torch.allclose(cov_b, cov[:, 0].double())

    if verbose: print(f'{"`gaussian_barycenter_diag`":50s} {"0-test":20s} {"success":20s}')


def gaussian_w2_scipy(m0,m1,Sigma0,Sigma1):
    """
    compute the quadratic Wasserstein distance between two Gaussians with means m0 and m1 and covariances Sigma0 and Sigma1
    author: Julie Delon
    taken from: https://github.com/judelo/gmmot/blob/0984edc826b113e35c3260b699a4ff49ab39d25f/python/gmmot.py#L63
    """
    Sigma00  = spl.sqrtm(Sigma0)
    Sigma010 = spl.sqrtm(Sigma00@Sigma1@Sigma00)
    d        = np.linalg.norm(m0-m1)**2+np.trace(Sigma0+Sigma1-2*Sigma010)
    return d


def gaussian_barycenter_w2_scipy(mu, Sigma, alpha, N):
    """
    Compute the W2 barycenter between several Gaussians
    mu has size Kxd, with K the number of Gaussians and d the space dimension
    Sigma has size Kxdxd
    author: Julie Delon
    taken from: https://github.com/judelo/gmmot/blob/0984edc826b113e35c3260b699a4ff49ab39d25f/python/gmmot.py#L85
    """
    K = mu.shape[0]  # number of Gaussians
    d = mu.shape[1]  # size of the space
    Sigman = np.eye(d, d)
    mun = np.zeros((1, d))
    cost = 0

    for n in range(N):
        Sigmandemi = spl.sqrtm(Sigman)
        T = np.zeros((d, d))
        for j in range(K):
            T += alpha[j] * spl.sqrtm(Sigmandemi @ Sigma[j, :, :] @ Sigmandemi)
        Sigman = T

    for j in range(K):
        mun += alpha[j] * mu[j, :]

    for j in range(K):
        cost += alpha[j] * gaussian_w2_scipy(mu[j, :], mun, Sigma[j, :, :], Sigman)

    return mun, Sigman, cost  # return the Gaussian Barycenter (mun,Sigman) and the total cost


def gmm_ot_scipy(pi0,pi1,mu0,mu1,S0,S1):
    """
    Compute the optimal transport map and cost between 2 gaussian mixtures
    author: Julie Delon
    taken from: https://github.com/judelo/gmmot/blob/0984edc826b113e35c3260b699a4ff49ab39d25f/python/gmmot.py#L116
    """
    # return the GW2 discrete map and the GW2 distance between two GMM
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    d  = mu0.shape[1]
    S0 = S0.reshape(K0,d,d)
    S1 = S1.reshape(K1,d,d)
    M  = np.zeros((K0,K1))
    # First we compute the distance matrix between all Gaussians pairwise
    for k in range(K0):
        for l in range(K1):
            M[k,l]  = gaussian_w2_scipy(mu0[k,:],mu1[l,:],S0[k,:,:],S1[l,:,:])
    # Then we compute the OT distance or OT map thanks to the OT library
    wstar     = ot.emd(pi0,pi1,M)         # discrete transport plan
    distGW2   = np.sum(wstar*M)
    return wstar, distGW2


@retry(stop_max_attempt_number=_MAX_ATTEMPTS)
def test_w2_gaussian_vs_scipy(verbose=False):
    mean1, cov1 = _rand_mean_cov((2,3), _DIM)
    mean2, cov2 = _rand_mean_cov((2,3), _DIM)
    res = w2_gaussian(mean1, mean2, cov1, cov2, make_pd=False, verbose=verbose)

    assert res.shape == torch.Size([2,3])
    for i in range(2):
        for j in range(3):
            scipy_res = gaussian_w2_scipy(
                mean1[i,j].double().numpy(),
                mean2[i,j].double().numpy(),
                cov1[i,j].double().numpy(),
                cov2[i,j].double().numpy()
            )
            assert torch.abs(scipy_res - res[i,j]) < STABILITY_CONST

    if verbose: print(f'{"`w2_gaussian` vs scipy":50s} {"value test":20s} {"success":20s}')

@retry(stop_max_attempt_number=_MAX_ATTEMPTS)
def test_w2_barycenter_vs_scipy(verbose=False):
    n_iter = 100
    mean, cov = _rand_mean_cov((2, 3), _DIM)
    var = torch.diagonal(cov, dim1=-1, dim2=-2)  # extract variance
    assert var.shape == mean.shape == torch.Size([2, 3, _DIM])

    weights = torch.randn(2, 3, dtype=torch.double).abs()
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # diagonal case
    mean_b, var_b = gaussian_barycenter(mean, var, weights, diag=True)

    for i in range(2):
        mean_scipy, cov_scipy, _ = gaussian_barycenter_w2_scipy(
            mean[i].double().numpy(),
            torch.diag_embed(var[i].double()).numpy(),
            weights[i].double().numpy(),
            N=n_iter
        )
        assert torch.allclose(torch.from_numpy(mean_scipy).double(), mean_b[i])
        assert torch.allclose(torch.from_numpy(cov_scipy).double().diagonal(), var_b[i])

    # full matrix case
    mean_b, cov_b = gaussian_barycenter(mean, cov, weights, diag=False, n_iter=n_iter)

    for i in range(2):
        mean_scipy, cov_scipy, _ = gaussian_barycenter_w2_scipy(
            mean[i].double().numpy(),
            cov[i].double().numpy(),
            weights[i].double().numpy(),
            N=n_iter
        )
        assert torch.allclose(torch.from_numpy(mean_scipy).double(), mean_b[i])
        assert torch.allclose(torch.from_numpy(cov_scipy).double(), cov_b[i])

    if verbose: print(f'{"`w2_barycenter` vs scipy":50s} {"value test":20s} {"success":20s}')

@retry(stop_max_attempt_number=_MAX_ATTEMPTS)
def test_sinkhorn_vs_pot(verbose=False):
    cost = torch.randn(2, 3, _DIM, _DIM).abs()
    cost = cost + cost.transpose(-1, -2)  # ensure cost matrix is symmetric
    a = torch.randn(2, 3, _DIM).abs()
    b = torch.randn(2, 3, _DIM).abs()
    a /= a.sum(dim=-1, keepdim=True)
    b /= b.sum(dim=-1, keepdim=True)
    a = a.double()
    b = b.double()
    cost = cost.double()
    pi = sinkhorn_log(a, b, cost, reg=1e-5, max_iter=1000, threshold=STABILITY_CONST)
    for i in range(2):
        for j in range(3):
            pi_pot = ot.bregman.sinkhorn(
                a[i,j], b[i,j], cost[i,j],
                reg=1e-5, method='sinkhorn_log', numItermax=1000,
                stopThr=STABILITY_CONST, warn=False
            )
            assert torch.allclose(pi[i,j], pi_pot)

    if verbose: print(f'{"`sinkhorn` vs POT":50s} {"value test":20s} {"success":20s}')

@retry(stop_max_delay=100000)
def test_ot_gmm_vs_scipy(verbose=False):
    mean1, cov1 = _rand_mean_cov([2, 10], _DIM)
    mean2, cov2 = _rand_mean_cov([2, 20], _DIM)
    weights1 = torch.randn(2, 10, dtype=torch.double).abs()
    weights2 = torch.randn(2, 20, dtype=torch.double).abs()
    weights1 = weights1 / weights1.sum(dim=-1, keepdim=True)
    weights2 = weights2 / weights2.sum(dim=-1, keepdim=True)
    var1 = cov1.diagonal(dim1=-1, dim2=-2)
    var2 = cov2.diagonal(dim1=-1, dim2=-2)

    # diagonal case
    cost, coupling = batch_ot_gmm(
        mean1, mean2, var1, var2, True, weights1, weights2, reg=1e-7
    )
    for i in range(2):
        coupling_scipy, cost_scipy = gmm_ot_scipy(
            weights1[i].double().numpy(),
            weights2[i].double().numpy(),
            mean1[i].double().numpy(),
            mean2[i].double().numpy(),
            torch.diag_embed(var1[i]).double().numpy(),
            torch.diag_embed(var2[i]).double().numpy()
        )
        assert torch.abs(cost_scipy - cost[i]) < 1.
        assert torch.allclose(torch.from_numpy(coupling_scipy), coupling[i], atol=1.)

    # full matrix case
    cost, coupling = batch_ot_gmm(
        mean1, mean2, cov1, cov2, False, weights1, weights2, reg=1e-7
    )

    for i in range(2):
        coupling_scipy, cost_scipy = gmm_ot_scipy(
            weights1[i].double().numpy(),
            weights2[i].double().numpy(),
            mean1[i].double().numpy(),
            mean2[i].double().numpy(),
            cov1[i].double().numpy(),
            cov2[i].double().numpy()
        )
        assert torch.abs(cost_scipy - cost[i]) < 1.
        assert torch.allclose(torch.from_numpy(coupling_scipy), coupling[i], atol=1.)

    if verbose: print(f'{"`ot GMM` vs scipy + POT":50s} {"value test":20s} {"success":20s}')

if __name__ == "__main__":
    verb = True

    test_w2_gaussian_same_yeilds_0(verb)
    test_batch_w2_same_yeilds_0(verb)
    test_ot_gmm_same_yeilds_0(verb)
    test_gaussian_barycenter_same(verb)

    if verb: print('*' * 100)

    test_w2_gaussian_vs_scipy(verb)
    test_w2_barycenter_vs_scipy(verb)
    test_sinkhorn_vs_pot(verb)
    test_ot_gmm_vs_scipy(verb)

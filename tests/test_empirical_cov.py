"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of a CI for empirical covariance computation

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from math import sqrt
import torch
from ot_vae_lightning.ot.matrix_utils import is_spd, STABILITY_CONST
from ot_vae_lightning.ot.w2_utils import w2_gaussian
from torch.distributions import MultivariateNormal
from ot_vae_lightning.utils import human_format
from torch.linalg import vector_norm as norm
from ot_vae_lightning.ot.w2_utils import mean_cov


def empirical_cov_computation(dim, n_samples, batch_size=100):

    mean = torch.randn(dim, dtype=torch.double)
    m = torch.randn(dim, dim, dtype=torch.double)
    cov = m @ m.T

    assert is_spd(cov, strict=True), "cov expected to be SPD"

    d = MultivariateNormal(mean, cov)
    print("num samples | mean error | cov error | squared W2")
    print("------------|------------|-----------|-----------")
    for n in n_samples:
        z = d.sample([n])
        empirical_mean_all = z.mean(0)
        empirical_cov_all = 1/n * ((z - empirical_mean_all).T @ (z - empirical_mean_all))
        mean_approx_err = norm(mean-empirical_mean_all) / norm(mean)
        cov_error = norm(cov.flatten() - empirical_cov_all.flatten()) / norm(cov.flatten())
        w2 = w2_gaussian(mean, empirical_mean_all, cov, empirical_cov_all, make_pd=True, verbose=False)
        ns = human_format(n)
        ns_b = ns + ' - all'
        print(f"{ns_b + ' ' * (len('num samples')-len(ns_b))} |    {'{:.2f}'.format(mean_approx_err)}    |    "
              f"{'{:.2f}'.format(cov_error)}   |   {'{:.2f}'.format(w2)}")

        empirical_mean = torch.zeros_like(mean)
        empirical_cov = torch.zeros_like(cov)
        n_obs = 0

        for b in range(n // batch_size):
            z_batch = z[b * batch_size:(b+1) * batch_size]
            empirical_mean += z_batch.sum(0)
            empirical_cov += z_batch.T @ z_batch
            n_obs += z_batch.size(0)

        empirical_mean, empirical_cov = mean_cov(empirical_mean, empirical_cov, n_obs)
        mean_approx_err = norm(mean - empirical_mean) / norm(mean)
        cov_error = norm(cov.flatten() - empirical_cov.flatten()) / norm(cov.flatten())
        w2 = w2_gaussian(mean, empirical_mean, cov, empirical_cov)
        ns_s = ' ' * len(ns) + ' - part'
        print(f"{ns_s + ' ' * (len('num samples')-len(ns_s))} |    {'{:.2f}'.format(mean_approx_err)}    |    "
              f"{'{:.2f}'.format(cov_error)}   |   {'{:.2f}'.format(w2)}")

        mean_diff = norm(empirical_mean_all - empirical_mean) / norm(empirical_mean_all)
        cov_diff = norm(empirical_cov_all.flatten() - empirical_cov.flatten()) / norm(empirical_cov_all.flatten())
        w2_diff = w2_gaussian(empirical_mean_all, empirical_mean, empirical_cov_all, empirical_cov)
        ns_d = ' ' * len(ns) + ' - diff'
        print(f"{ns_d + ' ' * (len('num samples') - len(ns_d))} |    {'{:.2f}'.format(mean_diff)}    |    "
              f"{'{:.2f}'.format(cov_diff)}   |   {'{:.2f}'.format(w2_diff)}")
        print("------------|------------|-----------|-----------")
        assert mean_diff < STABILITY_CONST and cov_diff < STABILITY_CONST and w2_diff < sqrt(STABILITY_CONST)


def test_empirical_cov_computation(n_samples=(int(1e4), int(1e5))):
    for dim in [64, 128, 256, 512]:
        empirical_cov_computation(dim, n_samples)


if __name__ == "__main__":
    test_empirical_cov_computation((int(1e4), int(1e5)))

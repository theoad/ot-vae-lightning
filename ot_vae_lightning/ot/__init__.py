from ot_vae_lightning.ot.gaussian_transport import GaussianTransport
from ot_vae_lightning.ot.transport_callback import LatentTransport


def compute_mean_cov(sum, sum_corr, num_obs):
    """Empirical computation of mean and covariance matrix"""
    mean = sum / num_obs
    cov = sum_corr / num_obs - mean.unsqueeze(1).mm(mean.unsqueeze(0))
    return mean, cov

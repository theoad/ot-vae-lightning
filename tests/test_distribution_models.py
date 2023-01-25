from typing import Union
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches
import mpl_toolkits.mplot3d.art3d as art3d

import numpy as np

import torch
import torch.distributions as D
from torch.linalg import norm
from torch.utils.data import DataLoader, TensorDataset
from ot_vae_lightning.ot.distribution_models.gaussian_model import GaussianModel
from ot_vae_lightning.ot.distribution_models.gassian_mixture_model import GaussianMixtureModel
from ot_vae_lightning.ot.distribution_models.codebook_model import CodebookModel, CategoricalEmbeddings
from ot_vae_lightning.ot.transport.gaussian_transport import GaussianTransport
from ot_vae_lightning.ot.transport.discrete_transport import DiscreteTransport
from ot_vae_lightning.ot.transport.gmm_transport import GMMTransport
from ot_vae_lightning.ot.matrix_utils import eye_like
from ot_vae_lightning.utils import camel2snake
from retrying import retry

_SAMPLE_SIZE = int(1e4)
_DIM = 32
_LEADING_SHAPE = (2,)
_TOL = 1e-1  # TODO: Make this lower !
_MAX_ATTEMPTS = 2
_N_COMP = 16

def _rand_mean_cov(*shape, diag=False):
    mean = torch.randn(*shape)
    cov = torch.randn(*shape, shape[-1])
    # ensure matrix is SPD
    cov = cov @ cov.transpose(-1, -2) / _DIM + eye_like(cov) * 1e-5
    if diag: cov = torch.diagonal(cov, dim1=-1, dim2=-2)  # extract variance
    return mean, cov

def _generate_samples(mean, cov, numel: int, mixture: Union[bool, torch.Tensor] = False):
    diag = mean.shape == cov.shape
    dist = D.Independent(D.Normal(mean, cov ** 0.5), 1) if diag else D.MultivariateNormal(mean, cov)
    if isinstance(mixture, torch.Tensor) or mixture:
        weights = mixture if isinstance(mixture, torch.Tensor) else torch.ones(*mean.shape[:-1]) / mean.shape[-1]
        mix = D.Categorical(probs=weights)
        dist = D.MixtureSameFamily(mix, dist)
    samples = dist.sample((numel,))
    samples = samples.permute(*list(range(1, mean.dim()-bool(mixture))), 0, mean.dim()-bool(mixture))
    return samples, dist


def _plot_mean_cov(ax, means, covariances=None, weights=None, color=None):
        if len(means.shape) == 1:
            if covariances is not None: assert len(covariances.shape) == 2
            means = means[None, ...]
            if covariances is not None: covariances = covariances[None, ...]
            if weights is not None: weights = weights[None, ...]
        zeros = () if weights is None else (torch.zeros_like(means[:, 0]),)
        ax.scatter(means[:, 0], means[:, 1], *zeros, color=color, s=1)
        if covariances is not None:
            for mean, cov in zip(means, covariances):
                v, w = np.linalg.eigh(cov)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                for m in [1, 1.25, 1.6]:
                    ell = mpl.patches.Ellipse(mean, v[0]*m, v[1]*m, 180. + angle, color=color, fill=weights is None)
                    ell.set_clip_box(ax.bbox)
                    ell.set_alpha(0.2)
                    ax.add_artist(ell)
                    if weights is not None: art3d.pathpatch_2d_to_3d(ell, z=0, zdir="z")
        if weights is not None:
            for mean, weight in zip(means, weights):
                ax.quiver(*mean, 0, 0, 0, weight, color=color, arrow_length_ratio=0.05)

def _plot(title, X1, X2, mean1, mean2, cov1=None, cov2=None, weights1=None, weights2=None, plot=False, show=True):
    if not plot: return
    fig = plt.figure()
    if weights1 is not None and weights2 is not None:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlabel('prob')
        ax.set_zlim(bottom=0, top=1)
        ax.scatter(X1[:, 0], X1[:, 1], torch.zeros_like(X1[:, 0]), s=1, color='red')  # plot the data points
        ax.scatter(X2[:, 0], X2[:, 1], torch.zeros_like(X2[:, 0]), s=1, color='blue')  # plot the data points
    else:
        ax = fig.add_subplot()
        ax.scatter(X1[:, 0], X1[:, 1], s=1, color='red')  # plot the data points
        ax.scatter(X2[:, 0], X2[:, 1], s=1, color='blue')  # plot the data points
    _plot_mean_cov(ax, mean1, cov1, weights1, color='red')
    _plot_mean_cov(ax, mean2, cov2, weights2, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(left=min(X1[:, 0].min(), X2[:, 0].min()), right=max(X1[:, 0].max(), X2[:, 0].max()))
    ax.set_ylim(bottom=min(X1[:, 1].min(), X2[:, 1].min()), top=max(X1[:, 1].max(), X2[:, 1].max()))
    plt.title(title)
    if show: plt.show()
    return ax


def _snr(approx, ref, dim=-1):
    return norm(approx - ref, dim=dim) / norm(approx, dim=dim) / norm(ref, dim=dim)


class _TensorDatasetDim(TensorDataset):
    def __init__(self, tensors: torch.Tensor, dim=0):
        super().__init__(tensors)
        self.dim = dim

    def __getitem__(self, index):
        return self.tensors[0].select(dim=self.dim, index=index).unsqueeze(self.dim)

    def __len__(self):
        return self.tensors[0].size(self.dim)


def _tensor_dl(samples: torch.Tensor, batch_size):
    ds = _TensorDatasetDim(samples.detach().requires_grad_(False), dim=-2)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=partial(torch.cat, dim=-2))  # noqa


def _model(verbose=False, diag=False, mixture=False, w2_cfg={}, mix_cfg={}, mode='fit'):
    size = *_LEADING_SHAPE, _DIM
    size_params = (*_LEADING_SHAPE, _N_COMP, _DIM) if mixture else size
    mean, cov = _rand_mean_cov(*size_params, diag=diag)
    kwargs = dict(w2_cfg={**w2_cfg, 'diag': diag}, dtype=torch.double)
    if mixture: kwargs['mixture_cfg'] = {'n_components': _N_COMP, **mix_cfg}
    samples, gt_dist = _generate_samples(mean, cov, _SAMPLE_SIZE, mixture=mixture)
    Model = GaussianMixtureModel if mixture else GaussianModel
    model_name = camel2snake(Model.__name__)
    batch_size = 100

    @retry(stop_max_attempt_number=_MAX_ATTEMPTS)
    def _fit():
        model = Model(*size, **kwargs)
        model.fit(samples)
        assert model.w2(gt_dist).max() < _TOL
        if verbose: print(f'{f"{model_name}":25s} {f"fit":15s} {f"`diag`={diag}":15s} {"success":10s}')

    @retry(stop_max_attempt_number=_MAX_ATTEMPTS)
    def _update():
        model = Model(*size, **kwargs, update_decay=None)
        dl = _tensor_dl(samples, batch_size=batch_size)
        for i, batch in enumerate(dl): model.update(batch)
        model.fit()
        assert model.w2(gt_dist).max() < _TOL
        if verbose: print(f'{f"{model_name}":25s} {f"update":15s} {f"`diag`={diag}":15s} {"success":10s}')

    @retry(stop_max_attempt_number=_MAX_ATTEMPTS)
    def _autograd():
        epoch = 10
        model = Model(*size, **kwargs, update_with_autograd=True)
        dl = _tensor_dl(samples, batch_size=batch_size)
        optim = torch.optim.AdamW(model.parameters(), lr=0.1, betas=(0., 0.99), weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epoch * _SAMPLE_SIZE // batch_size, eta_min=1e-5)
        for _ in range(epoch):
            for i, batch in enumerate(dl):
                optim.zero_grad()
                nll = -model(batch).mean()
                nll.backward()
                optim.step()
                sched.step()
        assert model.w2(gt_dist).max() < _TOL
        if verbose: print(f'{f"{model_name}":25s} {f"autograd":15s} {f"`diag`={diag}":15s} {"success":10s}')

    if 'fit' in mode: _fit()
    if 'update' in mode: _update()
    if 'autograd' in mode: _autograd()


def test_gaussian_model(verbose=False):
    for diag in [True, False]:
        for mode in ['fit', 'update', 'autograd']:
            _model(verbose, diag, False, {'diag': diag, 'verbose': verbose}, {}, mode)


def test_gaussian_mixture_model(verbose=False):
    for diag in [True, False]:
        for mode in ['fit', 'update', 'autograd']:
            _model(verb, diag, True, {'verbose': verbose}, {'training_mode': 'argmax', 'topk': None}, mode)
        for metric in ['cosine', 'euclidean']:
            for p in [1., 2., 0.1]:
                for mode in ['sample', 'mean', 'argmax']:
                    mixture_cfg = {'metric': metric, 'training_mode': mode, 'p': p, 'topk': None}
                    success = 'success'
                    try: _model(False, diag, True, {'verbose': verbose}, mixture_cfg, 'fit')
                    except: success = 'failure'
                    if verbose: print(f'{f"gaussian_mixture_model":25s} {f"fit {mode}":15s} {f"`diag`={diag}":15s}'
                                      f' {f"`metric`={metric} {p}":15s} {f"{success}":10s}')


@retry(stop_max_attempt_number=_MAX_ATTEMPTS)
def test_codebook_model(verbose=False, plot=False):
    lead, n_comp, dim = (_LEADING_SHAPE, _N_COMP, _DIM) if not plot else ((), 10, 2)
    size = *lead, dim
    size_params = (*lead, n_comp, dim)
    mean, cov = _rand_mean_cov(*size_params, diag=False)
    samples, gt_dist = _generate_samples(mean, cov, _SAMPLE_SIZE, mixture=True)

    true_codebook = mean
    weights = torch.ones(*lead, n_comp) / n_comp
    distribution = CategoricalEmbeddings(true_codebook, probs=weights)
    batch_size = 100
    mixture_cfg = {'n_components': n_comp, 'metric': 'euclidean', 'training_mode': 'argmax', 'p': 2., 'topk': None}
    title = f"{mixture_cfg['metric']}(p={mixture_cfg['p']}), {mixture_cfg['training_mode']}"
    model = CodebookModel(*size, mixture_cfg=mixture_cfg)
    dl = _tensor_dl(samples, batch_size=batch_size)
    for i, batch in enumerate(dl):
        model.update(batch); _plot(title, samples, model.codebook, true_codebook, plot=plot)
    model.fit(); _plot(title, samples, model.codebook, true_codebook, plot=plot)
    success = 'success' if model.w2(distribution).max() < _TOL else 'failure'
    if verbose: print(f'{f"codebook":25s} {f"update":15s} {f"{success}":10s}')

def _plot_gaussian_transport():
    gt = GaussianTransport(2)
    mean1, cov1 = _rand_mean_cov(2, diag=False)
    gt.source_model.mean.data.copy_(mean1)
    gt.source_model.cov = cov1
    mean2, cov2 = _rand_mean_cov(2, diag=False)
    gt.target_model.mean.data.copy_(mean2)
    gt.target_model.cov = cov2
    w2 = gt.compute()
    print('w2', w2)
    samples1, _ = _generate_samples(mean1, cov1, 20)
    transported = gt.transport(samples1)
    ax = _plot("gaussian transport", samples1, transported, mean1, mean2, cov1, cov2, plot=True, show=False)
    for s, t in zip(samples1, transported):
        ax.arrow(*s, *(t - s), width=0.001, head_width=0.02, length_includes_head=True, alpha=0.5)
    plt.show()

def _plot_mixture_transport(discrete=True):
    tr = DiscreteTransport if discrete else GMMTransport
    n_comp = 10 if discrete else 3
    tr = tr(2, source_cfg={'mixture_cfg': {'n_components': n_comp, 'kmeans_iter': 0}},
            target_cfg={'mixture_cfg': {'n_components': n_comp, 'kmeans_iter': 0}},
            transport_type='sample')
    tr.source_model._n_obs.copy_(torch.randint(size=(n_comp,), low=1, high=20))
    tr.target_model._n_obs.copy_(torch.randint(size=(n_comp,), low=1, high=20))
    if isinstance(tr, GMMTransport):
        tr.source_model.weights.copy_(torch.randn(n_comp).softmax(-1))
        tr.target_model.weights.copy_(torch.randn(n_comp).softmax(-1))
        tr.source_model.mean += 1
        tr.source_model.mean *= 3
        tr.target_model.mean -= 1
        tr.target_model.mean *= 3
    else:
        tr.source_model.codebook += 2
        tr.target_model.codebook -= 2
    tr.compute()
    tr_mat = tr.transport_matrix
    source = tr.source_model.codebook if discrete else tr.source_model.mean
    target = tr.target_model.codebook if discrete else tr.target_model.mean
    if discrete:
        ax = _plot("discrete transport", source, target, tr.source_distribution.embeddings,
                   tr.target_distribution.embeddings, None, None, tr.source_distribution.probs,
                   tr.target_distribution.probs,
                   plot=True, show=False)
    else:
        ax = _plot("gmm transport", source, target, tr.source_model.mean,
                   tr.target_model.mean, tr.source_model.cov, tr.target_model.cov,
                   tr.source_model.weights, tr.target_model.weights,
                   plot=True, show=False)
    for s, t in zip(*torch.chunk(tr_mat.nonzero(), 2, 1)):
        ax.quiver(source[s,0], source[s, 1], 0, target[t, 0] - source[s, 0], target[t, 1] - source[s, 1], 0,
                  color='black', arrow_length_ratio=0.005, alpha=min(10 * tr_mat[s,t].item(),1))
    plt.show()


if __name__ == "__main__":
    verb = True

    # _plot_gaussian_transport()
    # _plot_mixture_transport()
    # _plot_mixture_transport(False)
    test_gaussian_model(verb)
    # test_gaussian_mixture_model(verb)
    # test_codebook_model(verb, True)

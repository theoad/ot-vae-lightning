"""
************************************************************************************************************************

`PyTorch Lightning <https://www.pytorchlightning.ai/>`_ implementation of Wasserstein 2 optimal gaussian transport

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from torch import Tensor

from ot_vae_lightning.ot.transport.base import TransportOperator
from ot_vae_lightning.ot.w2_utils import W2Mixin
from ot_vae_lightning.ot.distribution_models.gaussian_model import GaussianModel

__all__ = ['GaussianTransport']


class GaussianTransport(TransportOperator, W2Mixin):
    r"""
    Computes the following transport operators according to eq. 17, 19 in [1]:

    .. math::

        (17) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{-0.5}_{s} (\Sigma^{0.5}_{s} \Sigma_{t}
         \Sigma^{0.5}_{s})^{0.5} \Sigma^{-0.5}_{s} + P/G^{*} I

        (19) T_{s \longrightarrow t} = (1 - P/G^{*}) \Sigma^{+0.5}_{t} (\Sigma^{0.5}_{t} \Sigma_{s}
         \Sigma^{0.5}_{t})^{0.5} \Sigma^{-0.5}_{t} \Sigma^{\dagger}_{s} + P/G^{*} I

         \Sigma_w = \Sigma^{0.5}_{t} (I - \Sigma^{0.5}_{t} T^{*} \Sigma^{\dagger}_{s} T^{*}
          \Sigma^{0.5}_{t}) \Sigma^{0.5}_{t},  T^{*} = T_{t \longrightarrow s} (17)

    [1] D. Freirich, T. Michaeli and R. Meir. A Theory of the Distortion-Perception Tradeoff in Wasserstein Space
    """

    def __init__(
            self, *size,
            source_cfg={},
            target_cfg={},
            transport_cfg={},
            **kwargs
    ):
        W2Mixin.__init__(self, **transport_cfg)
        TransportOperator.__init__(
            self, *size,
            source_model=GaussianModel(*size, w2_cfg=transport_cfg, **source_cfg),
            target_model=GaussianModel(*size, w2_cfg=transport_cfg, **target_cfg),
            **kwargs
        )

        self.transport_operator = None
        self.cov_stochastic_noise = None

    def reset(self) -> None:
        super().reset()
        self.transport_operator = None
        self.cov_stochastic_noise = None

    def compute(self) -> Tensor:
        self.fit_models()

        w2 = self.w2_gaussian(
            self.source_model.mean,
            self.target_model.mean,
            self.source_model.cov,
            self.target_model.cov,
        )

        self.transport_operator, self.cov_stochastic_noise = self.compute_transport_operators(
            self.source_model.cov,
            self.target_model.cov,
        )
        return w2

    def transport(self, inputs: Tensor) -> Tensor:
        if inputs.size(-1) != self.dim:
            raise ValueError("`inputs` dimensionality must match the model dimensionality")
        if inputs.shape[:-2] != self.leading_shape and inputs.shape[:-1] != self.leading_shape:
            raise ValueError("`inputs` leading dims must match the model batch_shape with optional trailing batch dimensions")
        is_batched = inputs.dim() == len(self.leading_shape) + 2

        transported = self.apply_transport(
            inputs,
            self.source_model.mean,
            self.target_model.mean,
            self.transport_operator,
            self.cov_stochastic_noise,
            batch_dim=-2 if is_batched else None
        )
        return transported.type_as(inputs)

    def extra_repr(self) -> str:
        return super().extra_repr() + W2Mixin.__repr__(self)

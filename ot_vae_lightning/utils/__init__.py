import torchvision.transforms as T
from ot_vae_lightning.utils.collage import Collage


class UnNormalize(T.Compose):
    def __init__(self, mean, std, inplace=False):
        super().__init__([
            T.Normalize([0.] * len(mean), [1/s for s in std], inplace),
            T.Normalize([-m for m in mean], [1.] * len(std), inplace),
        ])

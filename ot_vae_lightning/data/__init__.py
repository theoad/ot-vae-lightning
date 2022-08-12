import torchvision.transforms as T
from ot_vae_lightning.data.base import BaseDatamodule, dataset_split
from ot_vae_lightning.data.torchvision_datamodule import TorchvisionDatamodule
from ot_vae_lightning.utils import UnNormalize, ToTensor


class MNIST(TorchvisionDatamodule):
    _mean, _std = (0.1307,), (0.3081,)
    _normalize = T.Normalize(_mean, _std)
    _denormalize = UnNormalize(_mean, _std)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='MNIST',
            root='~/.cache',
            download=True,
            train_transform=T.Compose([ToTensor(), MNIST._normalize]),
            inference_preprocess=T.Compose([MNIST._normalize]),
            inference_postprocess=T.Compose([MNIST._denormalize])
        )


class MNIST32(TorchvisionDatamodule):
    _mean, _std = (0.1307,), (0.3081,)
    _normalize = T.Normalize(_mean, _std)
    _denormalize = UnNormalize(_mean, _std)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='MNIST',
            root='~/.cache',
            download=True,
            train_transform=T.Compose([ToTensor(), MNIST32._normalize, T.Pad(2)]),
            inference_preprocess=T.Compose([MNIST32._normalize, T.Pad(2)]),
            inference_postprocess=T.Compose([T.CenterCrop(28), MNIST32._denormalize])
        )


class CIFAR10(TorchvisionDatamodule):
    _mean, _std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    _normalize = T.Normalize(_mean, _std)
    _denormalize = UnNormalize(_mean, _std)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='CIFAR10',
            root='~/.cache',
            download=True,
            train_transform=T.Compose([T.RandomHorizontalFlip(), ToTensor(), CIFAR10._normalize]),
            inference_preprocess=T.Compose([CIFAR10._normalize]),
            inference_postprocess=T.Compose([CIFAR10._denormalize]),
        )


class ImageNet(TorchvisionDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageNet',
            root='~/data/ImageNet',
            download=False,
            train_transform=T.Compose([T.RandomHorizontalFlip(), ToTensor()]),
        )


class ImageNet256(TorchvisionDatamodule):
    _mean, _std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    _normalize = T.Normalize(_mean, _std)
    _denormalize = UnNormalize(_mean, _std)
    _resize = [T.Resize(256), ToTensor()]

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageNet',
            root='~/data/ImageNet',
            download=False,
            train_transform=T.Compose([T.RandomHorizontalFlip(), *ImageNet256._resize, ImageNet256._normalize]),
            val_transform=T.Compose(ImageNet256._resize),
            test_transform=T.Compose(ImageNet256._resize),
            inference_preprocess=T.Compose([ImageNet256._normalize]),
            inference_postprocess=T.Compose([ImageNet256._denormalize]),
        )


class ImageNet224(TorchvisionDatamodule):
    _mean, _std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    _normalize = T.Normalize(_mean, _std)
    _denormalize = UnNormalize(_mean, _std)
    _resize = [T.Resize(224), ToTensor()]

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageNet',
            root='~/data/ImageNet',
            download=False,
            train_transform=T.Compose([T.RandomHorizontalFlip(), *ImageNet224._resize, ImageNet224._normalize]),
            val_transform=T.Compose(ImageNet224._resize),
            test_transform=T.Compose(ImageNet224._resize),
            inference_preprocess=T.Compose([ImageNet224._normalize]),
            inference_postprocess=T.Compose([ImageNet224._denormalize]),
        )


class FFHQ128(TorchvisionDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageFolder',
            root=('~/data/ffhq_128_train', '~/data/ffhq_128_test'),
            download=False,
            train_transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]),
            test_transform=T.ToTensor()
        )

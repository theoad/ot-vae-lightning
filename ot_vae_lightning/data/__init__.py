import torchvision.transforms as T
from ot_vae_lightning.data.base import *
from ot_vae_lightning.data.torchvision_datamodule import *
from ot_vae_lightning.utils import UnNormalize, ToTensor


class MNIST(TorchvisionDatamodule):
    IMG_SIZE = (28, 28)
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
            inference_preprocess=MNIST._normalize,
            inference_postprocess=MNIST._denormalize
        )


class MNIST32(TorchvisionDatamodule):
    IMG_SIZE = (32, 32)
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
    IMG_SIZE = (32, 32)
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
            inference_preprocess=CIFAR10._normalize,
            inference_postprocess=CIFAR10._denormalize,
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
    IMG_SIZE = (256, 256)
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
            inference_preprocess=ImageNet256._normalize,
            inference_postprocess=ImageNet256._denormalize,
        )


class ImageNet224(TorchvisionDatamodule):
    IMG_SIZE = (224, 224)
    _mean, _std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    _normalize = T.Normalize(_mean, _std)
    _denormalize = UnNormalize(_mean, _std)
    _resize = [T.CenterCrop(IMG_SIZE), T.Resize(IMG_SIZE), ToTensor()]

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageNet',
            root='~/data/ImageNet',
            download=False,
            train_transform=T.Compose([T.RandomHorizontalFlip(), *ImageNet224._resize, ImageNet224._normalize]),
            val_transform=T.Compose(ImageNet224._resize),
            test_transform=T.Compose(ImageNet224._resize),
            inference_preprocess=ImageNet224._normalize,
            inference_postprocess=ImageNet224._denormalize,
        )


class FFHQ128(TorchvisionDatamodule):
    IMG_SIZE = (128, 128)
    _mean, _std = (0.5207, 0.4254, 0.3805), (0.1164, 0.1110, 0.1162)
    _normalize = T.Normalize(_mean, _std)
    _denormalize = UnNormalize(_mean, _std)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageFolder',
            root=('~/data/ffhq_128_train', '~/data/ffhq_128_test'),
            download=False,
            train_transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), FFHQ128._normalize]),
            inference_preprocess=FFHQ128._normalize,
            inference_postprocess=FFHQ128._denormalize,
        )


class FFHQ64(TorchvisionDatamodule):
    IMG_SIZE = (64, 64)
    _mean, _std = (0.5207, 0.4254, 0.3805), (0.1164, 0.1110, 0.1162)
    _normalize = T.Normalize(_mean, _std)
    _denormalize = UnNormalize(_mean, _std)
    _resize = [T.Resize(64), ToTensor()]

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageFolder',
            root=('~/data/ffhq_128_train', '~/data/ffhq_128_test'),
            download=False,
            train_transform=T.Compose([T.RandomHorizontalFlip(), *FFHQ64._resize, FFHQ64._normalize]),
            val_transform=T.Compose(FFHQ64._resize),
            test_transform=T.Compose(FFHQ64._resize),
            inference_preprocess=FFHQ64._normalize,
            inference_postprocess=FFHQ64._denormalize,
        )

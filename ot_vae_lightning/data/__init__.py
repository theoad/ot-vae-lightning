import torchvision.transforms as T
from ot_vae_lightning.data.base import BaseDatamodule
from ot_vae_lightning.data.torchvision_datamodule import TorchvisionDatamodule


class MNISTDatamodule(TorchvisionDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='MNIST',
            root='~/.cache',
            download=True,
            train_transform=T.Compose([T.Pad(2), T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]),
            test_transform=T.Compose([T.Pad(2), T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        )


class CIFAR10Datamodule(TorchvisionDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='CIFAR10',
            root='~/.cache',
            download=True,
            train_transform=T.Compose([T.RandomHorizontalFlip(),
                                        T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]),
            test_transform=T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]),
        )


class ImageNetDatamodule(TorchvisionDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageNet',
            root='~/data/ImageNet',
            download=False,
            train_transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]),
            test_transform=T.ToTensor()
        )


class ImageNet256Datamodule(TorchvisionDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageNet',
            root='~/data/ImageNet',
            download=False,
            train_transform=T.Compose([
                T.RandomHorizontalFlip(), T.Resize(256), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            test_transform=T.Compose([
                T.Resize(256), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        )


class ImageNet224Datamodule(TorchvisionDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageNet',
            root='~/data/ImageNet',
            download=False,
            train_transform=T.Compose([
                T.RandomHorizontalFlip(), T.Resize(224), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            test_transform=T.Compose([
                T.Resize(224), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        )


class FFHQ128Datamodule(TorchvisionDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            dataset_name='ImageFolder',
            root=('~/data/thumbnails128x128_train', '~/data/thumbnails128x128_test'),
            train_transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]),
            test_transform=T.ToTensor()
        )

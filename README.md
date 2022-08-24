---

<div align="center">    
 
# Optimal Transport VAE   

![CI testing](https://github.com/theoad/ot-vae-lightning/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
Official implementation of Optimal Transport VAE in Lightning

## How to run   
First, install dependencies   
```bash
# For M1 silicon, uncomment the followin:
# export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
# export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

pip install git+https://github.com/theoad/ot-vae-lightning
```

Run training
```bash
python model/vae.py \
--config configs/trainer.yaml \
--config configs/wandb.yaml \
--config configs/vanilla_vae.yaml
```

## Usage

### Training using pytorch-lightning Trainer

```python
import torch
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning import Trainer, seed_everything

from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio

from ot_vae_lightning.model import VAE
from ot_vae_lightning.prior import GaussianPrior
from ot_vae_lightning.data import MNIST32
from ot_vae_lightning.networks import CNN

if __name__ == "__main__":
    seed_everything(42)

    trainer = Trainer(max_epochs=10, callbacks=RichProgressBar())
    datamodule = MNIST32(train_batch_size=250)

    in_channels, in_resolution = 1, 32  # MNISTDatamodule pads MNIST images such that the resolution is a power of 2
    latent_channels, latent_resolution = 128, 1  # latent vectors will have shape [128, 1, 1]

    encoder = CNN(  # Simple nn.Module
        in_channels,
        latent_channels * 2,  # must double the number of channels in the encoder to allow re-parametrization trick
        in_resolution,
        latent_resolution,
        capacity=8,
        down_sample=True
    )

    decoder = CNN(  # Simple nn.Module
        latent_channels,
        in_channels,
        latent_resolution,
        in_resolution,
        capacity=8,
        up_sample=True
    )

    model = VAE(  # LightningModule
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=encoder,
        decoder=decoder,
        prior=GaussianPrior(loss_coeff=0.1),
    )

    assert model.latent_size == torch.Size((latent_channels, latent_resolution, latent_resolution))

    # Train
    trainer.fit(model, datamodule)
    trainer.save_checkpoint("vanilla_vae.ckpt")

    # Test
    model.freeze()
    results = trainer.test(model, datamodule)
    assert results[0]['test/metrics/psnr'] > 14
```

### Inference using the lightning model

```python
import torch
from torchvision.datasets import MNIST
from ot_vae_lightning.model import VAE
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

# Inference
vae = VAE.load_from_checkpoint("vanilla_vae.ckpt")
vae.freeze()  # put model in eval automatically

# The pre/post-processing transforms from the training datamodule are automatically loaded with the checkpoint
# Use this flag to wrap user methods (forward, encode, decode) with appropriate pre/post-processing:
# - normalize images before inputting to the model
# - de-normalize model outputs
vae.inference = True

with torch.no_grad():
    x = torch.randn(10, 1, 28, 28)
    z = vae.encode(x)  # pre-processing is done implicitly
    assert z.shape == torch.Size((10, 128, 1, 1))

    samples = vae.sample(batch_size=5)  # post-processing is done implicitly
    assert samples.shape == torch.Size((5, 1, 28, 28))

    x_hat = vae(x)  # pre-processing and post-processing are done implicitly
    assert x_hat.shape == torch.Size((10, 1, 28, 28))

# Inference in production. No transforms tailored to the pretrained model. Just raw data !
raw_mnist = MNIST(
    "~/.cache",
    train=False,
    transform=T.ToTensor(),
    download=True
)
dl = DataLoader(raw_mnist, batch_size=250, shuffle=False)
trainer = Trainer(gpus=..., strategy=..., )  # Use lightning trainer to have powerful distributed inference
predictions = trainer.predict(vae, dl)
assert predictions[0].shape == torch.Size((250, 1, 28, 28))  # type: ignore[arg-type]
```

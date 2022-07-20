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
python model/vae.py fit --config configs/vanilla_vae.yaml --trainer.gpus 1
```

## Usage

### Training using pytorch-lightning Trainer
```python
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning import Trainer, seed_everything
from ot_vae_lightning.model import VAE
from ot_vae_lightning.prior import GaussianPrior
from ot_vae_lightning.data import MNISTDatamodule
from ot_vae_lightning.networks import AutoEncoder
from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio


if __name__ == "__main__":
    seed_everything(42)

    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        autoencoder=AutoEncoder(1, 128, True, 32, 1, None, 8, 2, True, "batchnorm", "relu"),
        prior=GaussianPrior(loss_coeff=0.1)
    )

    trainer = Trainer(max_epochs=2, callbacks=RichProgressBar())
    datamodule = MNISTDatamodule(train_batch_size=40)

    trainer.fit(model, datamodule)
    trainer.save_checkpoint("vanilla_vae_mnist.ckpt", weights_only=True)

    results = trainer.test(model, datamodule)
    assert results[0]['test/psnr_epoch'] > 17
```

### Inference using the lightning model
```python
import torch
from ot_vae_lightning.model import VAE
from ot_vae_lightning.data import MNISTDatamodule

vae = VAE.load_from_checkpoint("vanilla_vae_mnist.ckpt")
vae.eval()

data = MNISTDatamodule()
preprocess = data.test_transform

x = torch.randn(10, 1, 28, 28)

with torch.no_grad():
    x_hat = vae(preprocess(x))

samples = vae.samples(10)
```

### Inference using the nn.Modules
```python
import torch
from ot_vae_lightning.networks import AutoEncoder
from ot_vae_lightning.data import MNISTDatamodule
from ot_vae_lightning.prior import GaussianPrior

# create the PyTorch model and load the checkpoint weights
checkpoint = torch.load("vanilla_vae_mnist.ckpt")
hyper_parameters = checkpoint["hyper_parameters"]

# if you want to restore any hyperparameters, you can pass them too
model = AutoEncoder(**hyper_parameters)
prior = GaussianPrior()
state_dict = checkpoint["state_dict"]

# update keys by dropping `auto_encoder.`
for key in state_dict.keys():
    state_dict[key.replace("autoencoder.", "")] = state_dict.pop(key)

model.load_state_dict(state_dict)
model.eval()

data = MNISTDatamodule()
preprocess = data.test_transform

x = torch.randn(10, 1, 28, 28)

with torch.no_grad():
    x_hat = model(preprocess(x))
```

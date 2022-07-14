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

### Usage
```python
from pytorch_lightning import Trainer, seed_everything
from ot_vae_lightning.model import VAE
from ot_vae_lightning.prior import GaussianPrior
from ot_vae_lightning.data import MNISTDatamodule
from ot_vae_lightning.networks import CNN
from torchmetrics import MetricCollection
from torchmetrics.image.psnr import PeakSignalNoiseRatio


if __name__ == "__main__":
    seed_everything(42)

    model = VAE(
        metrics=MetricCollection({'psnr': PeakSignalNoiseRatio()}),
        encoder=CNN(1, 128, 32, 2, None, 4, 2, True, False),
        decoder=CNN(64, 1, 2, 32, None, 4, 2, False, True),
        prior=GaussianPrior()
    )

    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    datamodule = MNISTDatamodule(train_batch_size=40)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
```   

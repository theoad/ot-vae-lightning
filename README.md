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
python model/vae.py fit --config configs/vanilla_vae.yaml --trainer.gpus 1 --optimizer=Adam --optimizer.lr=0.001
```

### Usage
```python

```   

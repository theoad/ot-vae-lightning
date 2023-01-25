#!/bin/sh
#accelerate launch networks/autoencoder.py
python model/vae.py -c configs/vae/defaults.yaml -c configs/vae/defaults_imagenet.yaml -c configs/ddp.yaml

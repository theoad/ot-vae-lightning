#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='ot_vae_lightning',
    version='0.0.1',
    description='Implementation of Optimal Transport VAE',
    author='Theo J. Adrai',
    author_email='tjtadrai@gmail.com',
    url='https://github.com/theoad/ot-vae-lightning',
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'pytorch-lightning',
        'jsonargparse[signatures]',
        'rich',
        'torchmetrics',
        'torch_fidelity',
        'numpy',
        'sympy'
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

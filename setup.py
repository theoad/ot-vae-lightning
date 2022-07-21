#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ot_vae_lightning',
    version='0.1.0',
    description='Implementation of Optimal Transport VAE',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Theo J. Adrai',
    author_email='tjtadrai@gmail.com',
    url='https://github.com/theoad/ot-vae-lightning',
    install_requires=requirements,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

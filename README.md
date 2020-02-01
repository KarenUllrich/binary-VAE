##  Binary VAE with BinConcrete latent distribution

Tensorflow implmentation of a VAE with Bininary Concrete (BinConcrete) latent distribution, based on:

["The concrete distribution: A continuous relaxation of discrete random variables"](https://arxiv.org/pdf/1611.00712.pdf)
Maddison, Chris J., Andriy Mnih, and Yee Whye Teh, ICLR, 2017


You can simply run this code by

	python experiment.py

### Requirements

This code has been tested with
-   `python 3.6`
-   `tensorflow 2.1.0` 
-   `tensorflow-probability 0.9.0` 
-   `matplotlib 3.1.2` 


Install conda environment via


	conda env create -f environment.yml 
	source activate binary_vae


### Maintenance

Please be warned that this repository is not going to be maintained regularly.

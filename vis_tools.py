#!/usr/bin/env python3
"""Visualization tools

Author, Karen Ullrich June 2019
"""

import matplotlib.pyplot as plt


def plot_samples(model):
    samples = model.decode(0., model.prior_sample_fun(10, ), return_probs=True)
    fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(10, 1))
    fig.suptitle("Reconstructions of 10 random vectors of the latent space.")
    for idx, sample in enumerate(samples):
        ax[idx].imshow(sample[:, :, 0], cmap='gray')
        ax[idx].axis('off')
    plt.show()


def plot_reconstuctions(model, dataset):
    # get 10 random images from the dataset
    x = dataset.__iter__().next()[:10]
    # reconstruct these images
    z, _, _ = model.encode(x)
    reconstructions = model.decode(x, z, return_probs=True)
    fig, ax = plt.subplots(nrows=2, ncols=10, figsize=(10, 2))
    fig.suptitle("Reconstructions of 10 random data point")
    for idx, (recon, data) in enumerate(zip(reconstructions, x)):
        ax[0, idx].imshow(data[:, :, 0], cmap='gray')
        ax[0, idx].axis('off')
        ax[1, idx].imshow(recon[:, :, 0], cmap='gray')
        ax[1, idx].axis('off')
    plt.show()

#!/usr/bin/env python3
"""Variational Autoencoder models.

Available latent distributions:
    * Gaussian/ Normal [1]
    * RelaxedBernoulli/ BinConcrete [2]

[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
[2] Maddison, Chris J., Andriy Mnih, and Yee Whye Teh. "The concrete
    distribution: A continuous relaxation of discrete random variables."

Author, Karen Ullrich June 2019
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # This is heavily inspired by an official VAE tutorial,
        # https://www.tensorflow.org/beta/tutorials/generative/cvae
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2),
                    activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2),
                    activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(400, activation='relu'),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim)
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(400, activation='relu'),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    def encode(self, x):
        pass

    def decode(self, x, z, return_probs=False):
        logits = self.generative_net(z)

        # Bernoullli observation model is equivalent to cross entropy loss
        observation_dist = tfd.Bernoulli(logits=logits)
        if return_probs:
            return observation_dist.probs
        logpx_z = tf.reduce_sum(observation_dist.log_prob(x), axis=[1, 2, 3])
        return logpx_z


class BVAE(VAE):
    def __init__(self, latent_dim, prior_temperature=0.1):
        super().__init__(latent_dim)

        probs = 0.5 * tf.ones(latent_dim)
        self.prior = tfd.Logistic(tf.log(probs) / prior_temperature,
                                  1. / prior_temperature)
        self.prior_sample_fun = lambda x: tf.sigmoid(self.prior.sample(x))

    def encode(self, x, temperature=0.5):
        logits, _ = tf.split(self.inference_net(x), num_or_size_splits=2,
                             axis=1)
        # we use
        latent_dist = tfd.Logistic(logits / temperature, 1. / temperature)
        # instead of
        # tfd.RelaxedBernoulli(temperature=temperature, logits=logits)
        # otherwise we run into underflow issues when computing the log_prob

        logistic_samples = latent_dist.sample()
        return tf.sigmoid(logistic_samples), latent_dist.log_prob(
            logistic_samples), self.prior.log_prob(
            logistic_samples)


class NVAE(VAE):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)
        self.prior = tfd.Normal(loc=tf.zeros(latent_dim),
                                scale=tf.ones(latent_dim))
        self.prior_sample_fun = self.prior.sample

    def encode(self, x):
        loc, logvar = tf.split(self.inference_net(x), num_or_size_splits=2,
                               axis=1)
        latent_dist = tfd.Normal(loc=loc, scale=tf.exp(logvar))
        latent_samples = latent_dist.sample()
        return latent_samples, latent_dist.log_prob(
            latent_samples), self.prior.log_prob(latent_samples)

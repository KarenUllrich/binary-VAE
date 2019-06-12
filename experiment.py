#!/usr/bin/env python3
"""Training of a VAE

This is heavily inspired by an official VAE tutorial,
https://www.tensorflow.org/beta/tutorials/generative/cvae

Author, Karen Ullrich June 2019
"""

from absl import app
import tensorflow as tf
import tensorflow_probability as tfp
import time

from model import BVAE
from model import NVAE
from vis_tools import plot_reconstuctions
from vis_tools import plot_samples

tfd = tfp.distributions
tf.enable_eager_execution()


def get_dataset():
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28,
                                        1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype(
        'float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    # Binarization
    # Note that this binarization is only a quick hack, it is wiser to use
    # dynamically binarized or statically binarized MNIST. Elbo scores with this
    # dataset are going to be better than one would expect
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        60000).batch(FLAGS.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(
        10000).batch(FLAGS.batch_size)
    return train_dataset, test_dataset


def compute_loss(model, x):
    # z ~ q(z|x), q(z|x), p(z)
    z, logqz_x, logpz = model.encode(x)
    # p(x|z)
    logpx_z = model.decode(x, z)

    # empirical KL divergence
    kl_div = tf.reduce_sum(logqz_x - logpz, axis=-1)

    # we minimize the negative evidence lower bound (nelbo)
    nelbo = - tf.reduce_mean(logpx_z - kl_div)
    return nelbo


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


def train(epochs, model, optimizer, train_dataset, test_dataset):
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            gradients, loss = compute_gradients(model, train_x)
            apply_gradients(optimizer, gradients, model.trainable_variables)
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current '
                  'epoch {}'.format(epoch, elbo, end_time - start_time))


def main(unused_argv):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    if FLAGS.latent_dist == 'normal':
        model = NVAE(latent_dim=FLAGS.latent_dim)
    elif FLAGS.latent_dist == 'binary':
        model = BVAE(latent_dim=FLAGS.latent_dim)
    train_dataset, test_dataset = get_dataset()
    train(FLAGS.num_epochs, model, optimizer, train_dataset, test_dataset)
    if FLAGS.visualize:
        plot_samples(model)
        plot_reconstuctions(model, test_dataset)


if __name__ == "__main__":
    flags = app.flags
    FLAGS = app.flags.FLAGS

    flags.DEFINE_float('learning_rate', 1e-4,
                       'Learning rate to use for training.')
    flags.DEFINE_integer('num_epochs', 5,
                         'Number of epochs to run training for.')
    flags.DEFINE_integer('latent_dim', 160, 'Number of latent distributions.')
    flags.DEFINE_integer('batch_size', 128,
                         'Batch size for training and evaluation.')
    flags.DEFINE_string('latent_dist', 'binary',
                        'The latent distribution may be BinConcrete (binary) or'
                        ' Gaussian (normal).')
    flags.DEFINE_boolean('visualize', True,
                         'Whether to plot decoder samples of random vectors at '
                         'the end of training.')

    app.run(main)

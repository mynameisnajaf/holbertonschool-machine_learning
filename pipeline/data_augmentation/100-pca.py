#!/usr/bin/env python3
"""PCA color augmentation (AlexNet style)"""

import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.
    """

    image = tf.cast(image, tf.float32)
    flat = tf.reshape(image, [-1, 3])

    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean

    cov = tf.matmul(centered, centered, transpose_a=True)
    cov /= tf.cast(tf.shape(flat)[0], tf.float32)

    eigvals, eigvecs = tf.linalg.eigh(cov)

    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)

    alphas = tf.cast(alphas, tf.float32)

    noise = tf.matmul(
        eigvecs,
        tf.reshape(eigvals * alphas, (3, 1))
    )

    noise = tf.reshape(noise, (1, 3))

    flat = flat + noise
    return tf.reshape(flat, tf.shape(image))

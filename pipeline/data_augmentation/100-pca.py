#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def pca_color(image, alphas):
    """PCA color augmentation"""
    orig_shape = tf.shape(image)
    flat_image = tf.reshape(image, [-1, 3])

    flat_image = tf.cast(flat_image, tf.float32)

    mean = tf.reduce_mean(flat_image, axis=0)
    centered = flat_image - mean

    cov = tf.matmul(centered, centered, transpose_a=True)
    cov /= tf.cast(tf.shape(centered)[0] - 1, tf.float32)

    eigvals, eigvecs = tf.linalg.eigh(cov)

    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)

    delta = tf.matmul(
        eigvecs,
        tf.reshape(alphas * tf.sqrt(eigvals), [-1, 1])
    )
    augmented = flat_image + tf.reshape(delta, [1, 3])
    augmented = tf.reshape(augmented, orig_shape)

    return augmented

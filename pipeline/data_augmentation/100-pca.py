#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def pca_color(image, alphas):
    """Do a PCA on the image"""
    img_float = tf.cast(image, tf.float32)

    pixels = tf.reshape(img_float, [-1, 3])

    num_samples = tf.cast(tf.shape(pixels)[0], tf.float32)
    covariance = tf.matmul(tf.transpose(pixels), pixels) / num_samples

    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)

    alphas = tf.cast(alphas, tf.float32)
    delta = tf.matmul(eigenvectors, tf.reshape(alphas * eigenvalues, (3, 1)))

    delta = tf.reshape(delta, (1, 1, 3))

    result = img_float + delta

    return tf.cast(tf.math.round(result), image.dtype)


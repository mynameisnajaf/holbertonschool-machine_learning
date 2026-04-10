#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def pca_color(image, alphas):
    """Do a PCA on the image"""
    img_float = tf.cast(image, tf.float32)
    image_size = tf.shape(image)[0] * tf.shape(image)[1]

    pixels = tf.reshape(img_float, [-1, 3])

    means = tf.reduce_mean(pixels, axis=0)
    centered = pixels - means

    num_samples = tf.cast(image_size, tf.float32)
    covariance = tf.matmul(tf.transpose(centered), centered) / (num_samples - 1.0)

    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)

    alphas = tf.cast(alphas, tf.float32)
    delta = tf.matmul(eigenvectors, tf.reshape(alphas * eigenvalues, (3, 1)))

    delta = tf.reshape(delta, (1, 1, 3))

    return image + tf.cast(tf.math.round(delta), image.dtype)

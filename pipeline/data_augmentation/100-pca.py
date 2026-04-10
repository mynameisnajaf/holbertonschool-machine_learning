#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def pca_color(image, alphas):
    """Do a PCA on the image"""
    img_float = tf.cast(image, tf.float32)

    pixels = tf.reshape(img_float, [-1, 3])

    means = tf.reduce_mean(pixels, axis=0)
    centered = pixels - means

    num_samples = tf.cast(tf.shape(pixels)[0], tf.float32)
    covariance = tf.matmul(tf.transpose(centered), centered) / (num_samples - 1.0)

    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)

    alphas = tf.cast(alphas, tf.float32)

    delta = tf.matmul(eigenvectors, tf.reshape(alphas * eigenvalues, (3, 1)))

    delta = tf.reshape(delta, (1, 1, 3))

    return tf.cast(tf.clip_by_value(tf.math.round(img_float + delta), 0, 255), image.dtype)

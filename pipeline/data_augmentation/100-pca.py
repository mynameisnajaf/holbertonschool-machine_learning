#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def pca_color(image, alphas):
    """Do a PCA on the image"""
    orig_type = image.dtype
    img_float = tf.cast(image, tf.float32)

    pixels = tf.reshape(img_float, [-1, 3])

    means = tf.reduce_mean(pixels, axis=0)
    centered_pixels = pixels - means

    num_samples = tf.cast(tf.shape(pixels)[0], tf.float32)
    covariance = tf.matmul(tf.transpose(centered_pixels), centered_pixels) / (num_samples - 1.0)

    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)

    alphas = tf.cast(alphas, tf.float32)
    delta = tf.matmul(eigenvectors, tf.reshape(alphas * eigenvalues, (3, 1)))

    delta = tf.reshape(delta, (1, 1, 3))
    result = img_float + delta

    return tf.cast(tf.clip_by_value(result, 0, 255), orig_type)

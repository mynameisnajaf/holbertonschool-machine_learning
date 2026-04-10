#!/usr/bin/env python3
"""PCA color augmentation (AlexNet style)"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: tf.Tensor (H, W, 3)
        alphas: tuple of length 3

    Returns:
        Augmented image (tf.Tensor)
    """

    image = tf.cast(image, tf.float32)

    # reshape image to (N, 3)
    flat = tf.reshape(image, [-1, 3])

    # compute mean
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean

    # covariance matrix (3x3)
    cov = tf.matmul(centered, centered, transpose_a=True)
    cov /= tf.cast(tf.shape(flat)[0], tf.float32)

    # eigenvalues + eigenvectors (symmetric matrix → use eigh)
    eigvals, eigvecs = tf.linalg.eigh(cov)

    # sort eigenvalues descending (important for AlexNet style)
    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)

    # convert alphas
    alphas = tf.cast(alphas, tf.float32)

    # PCA noise: eigvecs * (eigvals * alphas)
    noise = tf.matmul(
        eigvecs,
        tf.reshape(eigvals * alphas, (3, 1))
    )

    noise = tf.reshape(noise, (1, 3))

    # add noise
    flat = flat + noise

    # reshape back
    return tf.reshape(flat, tf.shape(image))

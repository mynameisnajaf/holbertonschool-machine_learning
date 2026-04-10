#!/usr/bin/env python3
"""PCA color augmentation"""
import tensorflow as tf


def pca_color(image, alphas):
    """Applies PCA color augmentation to image"""

    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.cast(image, tf.float32)

    # flatten image (H*W, 3)
    flat = tf.reshape(image, [-1, 3])

    # center
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean

    # covariance matrix
    cov = tf.matmul(centered, centered, transpose_a=True)
    cov /= tf.cast(tf.shape(flat)[0], tf.float32)

    # eigenvalues & eigenvectors
    eigvals, eigvecs = tf.linalg.eigh(cov)

    # sort (important!)
    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)

    # convert alphas
    alphas = tf.cast(alphas, tf.float32)

    # PCA noise
    noise = tf.tensordot(eigvecs, eigvals * alphas, axes=1)

    # reshape noise and apply
    noise = tf.reshape(noise, (1, 1, 3))
    image = image + noise

    return image

#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def pca_color(image, alphas):
    """PCA color augmentation using TensorFlow only"""

    # Convert image to float32 and normalize
    image = tf.cast(image, tf.float32) / 255.0

    # Reshape to (num_pixels, 3)
    flat = tf.reshape(image, [-1, 3])

    # Center the data
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean

    # Covariance matrix
    cov = tf.matmul(centered, centered, transpose_a=True)
    cov /= tf.cast(tf.shape(flat)[0] - 1, tf.float32)

    # Eigen decomposition
    eigvals, eigvecs = tf.linalg.eigh(cov)

    # Sort in descending order
    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)

    # Compute PCA noise
    alphas = tf.cast(alphas, tf.float32)
    delta = tf.matmul(
        eigvecs,
        tf.reshape(alphas * eigvals, [-1, 1])
    )

    # Add noise
    flat_aug = flat + tf.reshape(delta, [1, 3])

    # Reshape back
    augmented = tf.reshape(flat_aug, tf.shape(image))

    # Scale back to [0,255]
    augmented = tf.clip_by_value(augmented * 255.0, 0, 255)

    return tf.cast(augmented, tf.uint8)

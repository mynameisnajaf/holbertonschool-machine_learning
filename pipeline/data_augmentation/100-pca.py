#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def pca_color(image, alphas):
    """PCA color augmentation using TensorFlow only"""

    # Convert image to array and normalize
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.cast(image, tf.float32) / 255.0

    # Reshape to (num_pixels, 3)
    flat = tf.reshape(image, [-1, 3])

    # Compute mean and center data
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean

    # Covariance matrix
    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flat)[0], tf.float32)

    # Eigen decomposition
    eigvals, eigvecs = tf.linalg.eigh(cov)

    # Convert alphas to tensor
    alphas = tf.cast(alphas, tf.float32)

    # Compute PCA noise
    delta = tf.matmul(
        eigvecs,
        tf.expand_dims(alphas * eigvals, axis=1)
    )

    delta = tf.reshape(delta, [3])

    # Add perturbation
    augmented = flat + delta

    # Reshape back to image
    augmented = tf.reshape(augmented, tf.shape(image))

    # Clip values to valid range
    augmented = tf.clip_by_value(augmented, 0.0, 1.0)

    return augmented

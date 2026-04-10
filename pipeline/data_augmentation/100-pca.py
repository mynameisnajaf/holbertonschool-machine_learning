#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """PCA color augmentation"""
    original_image = tf.keras.preprocessing.image.img_to_array(image)
    cp_original = original_image.astype(float).copy()

    original_image = original_image / 255.0
    img_rs = original_image.reshape(-1, 3)

    img_centered = img_rs - np.mean(img_rs, axis=0)

    img_cov = np.cov(img_centered, rowvar=False)

    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    m1 = np.column_stack((eig_vecs))
    m2 = np.zeros((3, 1))

    m2[:, 0] = alphas * eig_vals[:]
    add_vect = np.matrix(m1) * np.matrix(m2)

    # RGB
    for idx in range(3):
        cp_original[..., idx] += add_vect[idx]

    cp_original = np.clip(cp_original, 0.0, 255.0)
    cp_original = cp_original.astype(np.uint8)
    return cp_original

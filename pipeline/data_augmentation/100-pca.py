#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def pca_color(image, alphas):
    """PCA color augmentation using TensorFlow only"""
    image_nparray = tf.keras.preprocessing.image.img_to_array(image)
    img = image_nparray.reshape(-1, 3).astype(float)

    return img

#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Changes the contrast of the image"""
    return tf.image.random_contrast(image, lower=lower, upper=upper)

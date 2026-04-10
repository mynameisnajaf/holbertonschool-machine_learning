#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def crop_image(image, size):
    """Crops an image at the specified size"""
    return tf.image.random_crop(image, size)

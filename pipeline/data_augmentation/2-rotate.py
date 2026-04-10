#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def rotate_image(image):
    """Rotates an image"""
    return tf.image.rot90(image)

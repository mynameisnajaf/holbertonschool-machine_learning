#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Changes the brightness of the image"""
    return tf.image.random_brightness(image, max_delta)

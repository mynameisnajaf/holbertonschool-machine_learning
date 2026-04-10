#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def flip_image(image):
    """Flip the image horizontally"""
    return tf.image.flip_left_right(image)

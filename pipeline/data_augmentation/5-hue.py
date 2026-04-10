#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def change_hue(image, delta):
    """Change the hue of the image"""
    return tf.image.adjust_hue(image, delta)

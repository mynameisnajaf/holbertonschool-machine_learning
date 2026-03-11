#!/usr/bin/env python3
"""A module for supervised learning"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """One-hot encoding of labels"""
    ohe = K.utils.to_categorical(labels, num_classes=classes)
    return ohe

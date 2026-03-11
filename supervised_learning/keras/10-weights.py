#!/usr/bin/env python3
"""Main file"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save weights to a file"""
    if save_format == 'keras':
        network.save_weights(filename, save_format='keras')


def load_weights(network, filename):
    """Load weights from a file"""
    weights = network.load_weights(filename)

#!/usr/bin/env python3
"""Main file"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Main function"""
    return network.predict(data, verbose=verbose)

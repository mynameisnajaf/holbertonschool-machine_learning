#!/usr/bin/env python3
"""Main file"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Main function"""
    return network.evaluate(data, labels, verbose=verbose)

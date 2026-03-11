#!/usr/bin/env python3
"""Main file"""
import tensorflow.keras as K


def save_config(network, filename):
    """Save config to file"""
    network.save_config(filename)


def load_config(filename):
    """Load config from file"""
    return load_config(filename)

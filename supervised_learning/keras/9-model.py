#!/usr/bin/env python3
"""Main file"""
import tensorflow.keras as K


def save_model(network, filename):
    """Save model"""
    network.save(filename)


def load_model(filename):
    """Load model"""
    network = K.models.load_model(filename)
    return network

#!/usr/bin/env python3
"""Main file"""
import tensorflow.keras as K


def save_config(network, filename):
    """Save config to file"""
    model_json = network.to_json()
    with open(filename, "w") as f:
        f.write(model_json)
    return None


def load_config(filename):
    """Load config from file"""
    with open(filename, "r") as f:
        loaded_model_json = f.read()
    model = K.models.model_from_json(loaded_model_json)
    return model

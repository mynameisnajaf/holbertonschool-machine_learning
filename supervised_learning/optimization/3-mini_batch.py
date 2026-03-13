#!/usr/bin/env python3
"""A module that does the trick"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create a mini batch of data"""
    mini_batches = []
#!/usr/bin/env python3
"""A module that does the trick"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create a mini batch of data"""
    mini_batches = []
    m = X.shape[0]

    number_of_batches = int(m / batch_size)
    for i in range(number_of_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size

        X_batch = X[start_index:end_index]
        Y_batch = Y[start_index:end_index]

        mini_batches.append([X_batch, Y_batch])

    if m % batch_size != 0:
        X_batch = X[number_of_batches * batch_size:]
        Y_batch = Y[number_of_batches * batch_size:]
        mini_batches.append([X_batch, Y_batch])

    return mini_batches

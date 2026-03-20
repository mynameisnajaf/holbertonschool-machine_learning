#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    """Create a dropout layer"""
    dropout = tf.keras.layers.Dropout(keep_prob)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    tensor = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=dropout,
    )

    return tensor(prev)

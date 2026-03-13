#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer"""
    dense = tf.keras.layers.Dense(units=n,
                                  kernel_initiliazer=tf.keras.initializers.VarianceScaling(
                                      mode='fan_avg'
                                  ),
                                  use_bias=False,
                                  )(prev)
    bn = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,
        scale=True
    )(dense)

    output = tf.keras.layers.Activation(activation)(bn)

    return output

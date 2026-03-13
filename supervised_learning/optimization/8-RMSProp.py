#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Creates a RMSProp operation"""
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            weight_decay=beta2,
                                            momentum=epsilon)
    return optimizer

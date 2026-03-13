#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Creates a momentum op"""
    optimizer = tf.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer

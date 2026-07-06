#!/usr/bin/env python3
"""Self-attention"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Self attention class"""

    def __init__(self, units):
        """Initialize SelfAttention class"""
        if type(units) is not int:
            raise TypeError(
                "units must be int representing the number of hidden units")
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """Call self attention layer"""
        W = self.W(tf.expand_dims(s_prev, 1))
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W + U))
        weights = tf.nn.softmax(V, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
    
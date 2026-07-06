#!/usr/bin/env python3
"""Scaled Dot Product Attention"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """calculates the scaled dot product attention"""
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_dot = matmul_qk / tf.sqrt(dk)
    if mask is not None:
        scaled_dot += (mask * -1e9)
    weights = tf.nn.softmax(scaled_dot, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights

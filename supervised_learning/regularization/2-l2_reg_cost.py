#!/usr/bin/env python3
"""A module to implement regularization cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """L2 regularization cost"""
    l2_losses = model.losses

    if l2_losses:
        l2_loss = tf.add_n(l2_losses)
    else:
        l2_loss = 0

    return cost + l2_loss

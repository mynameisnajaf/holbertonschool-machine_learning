#!/usr/bin/env python3
"""A module to implement regularization cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """L2 regularization cost"""
    l2_losses = model.losses
    l2_losses_tensor = tf.stack(l2_losses)
    total_cost = cost + l2_losses_tensor

    return total_cost

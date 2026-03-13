#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """The learning rate decay"""
    lr = alpha / (1 + decay_rate * ( global_step // decay_step))
    return lr

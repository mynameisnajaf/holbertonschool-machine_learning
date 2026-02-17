#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def sensitivity(confusion):
    """A function that does the trick"""
    classes = confusion.shape[0]
    sensitivity = []
    for i in range(classes):
        true_positive = 0
        all_positive = 0
        for j in range(classes):
            if i == j:
                true_positive += confusion[i][j]
            all_positive += confusion[i][j]
        sensitivity.append(true_positive / all_positive)
    return np.asarray(sensitivity)

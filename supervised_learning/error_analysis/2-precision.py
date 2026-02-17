#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def precision(confusion):
    """A function that does the trick"""
    classes = confusion.shape[0]
    precise = []
    for i in range(classes):
        true_positive = 0
        all_positive = 0
        for j in range(classes):
            if i == j:
                true_positive += confusion[j, i]
            all_positive += confusion[j, i]
        precise.append(true_positive / all_positive)
    return np.asarray(precise)

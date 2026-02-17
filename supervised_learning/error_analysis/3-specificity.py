#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def specificity(confusion):
    """A function that does the trick"""
    classes = confusion.shape[0]
    specificity = []
    for aclass in range(classes):
        true_negative = 0
        all_negative = 0
        for i in range(classes):
            if i == aclass:
                continue
            for j in range(classes):
                if j != aclass:
                    true_negative += confusion[i, j]
                all_negative += confusion[i, j]
        specificity.append(true_negative / all_negative)
    return np.asarray(specificity)

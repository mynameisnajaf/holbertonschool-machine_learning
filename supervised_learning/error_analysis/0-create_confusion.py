#!/usr/bin/env python3

"""A module that does the trick"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """A function that does the trick"""
    classes = np.unique(np.concatenate((labels, logits)))
    num_classes = len(classes)
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_mat[i, j] = np.sum((labels == classes[i]) & (logits == classes[j]))

    return confusion_mat, classes

#!/usr/bin/env python3
"""Creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    labels: one-hot numpy.ndarray of shape (m, classes)
    logits: one-hot numpy.ndarray of shape (m, classes)

    Returns:
    confusion matrix of shape (classes, classes)
    """
    m, classes = labels.shape
    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)

    confusion = np.zeros((classes, classes))

    for i in range(m):
        confusion[true_classes[i], predicted_classes[i]] += 1

    return confusion

#!/usr/bin/env python3

"""A module that does the trick"""


def create_confusion_matrix(labels, logits):
    """A function that does the trick"""
    tp = tn = fp = fn = 0

    for i in range(len(labels)):
        if labels[i] == 1 and logits[i] == 1:
            tp += 1
        elif labels[i] == 0 and logits[i] == 0:
            fn += 1
        elif labels[i] == 0 and logits[i] == 1:
            fp += 1
        elif labels[i] == 1 and logits[i] == 0:
            tn += 1

    confusion_matrix = [[tp, fn], [fp, tn]]
    return confusion_matrix

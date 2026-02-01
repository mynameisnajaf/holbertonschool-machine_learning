#!/usr/bin/env python3

"""A module that does the trick"""
import numpy as np


def definiteness(matrix):
    """A function that does the trick"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if all(eigenvalues > 0):
        return 'Positive definite'
    elif all(eigenvalues >= 0):
        return 'Positive semi-definite'
    elif all(eigenvalues < 0):
        return 'Negative definite'
    elif all(eigenvalues <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'

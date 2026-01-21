#!/usr/bin/env python3

"""A module that does the trick"""


def matrix_shape(matrix):
    """A function that does the trick"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape

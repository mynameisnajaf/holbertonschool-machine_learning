#!/usr/bin/env python3

"""A module that does the trick"""


def minor(matrix):
    """A function that does the trick"""
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    h = len(matrix)
    if h is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != h:
            raise ValueError("matrix must be a square matrix")
    for col in range(h):
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        return minor

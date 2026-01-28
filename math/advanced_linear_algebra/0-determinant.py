#!/usr/bin/env python3

"""A module that does the trick"""


def determinant(matrix):
    """A function that does the trick"""
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    h = len(matrix)
    if h is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) is 0 and h is 1:
            return 1
        if len(row) != h:
            raise ValueError("matrix must be a square matrix")
    if h is 1:
        return matrix[0][0]
    if h is 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return ((a * d) - (b * c))
    det = 0
    for col in range(h):
        minor = [row[:col] + row[col+1:] for row in matrix[1:]]
        cofactor = ((-1) ** col) * matrix[0][col] * determinant(minor)
        det += cofactor
    return det

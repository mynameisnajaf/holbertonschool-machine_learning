#!/usr/bin/env python3

"""A module that does the trick"""


def determinant(matrix):
    """A function does the trick"""
    h = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
    if h == 0:
            return 1
    if h == 1:
        return matrix[0][0]
    if h == 2:
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


def minor(matrix):
    """A function that does the trick"""
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    h = len(matrix)
    if h == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if len(row) == 0 and h == 1:
            return 1
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != h or len(row) == 0:
            raise ValueError("matrix must be a non-empty square matrix")
    minor_matrix = []
    for i in range(h):
        row_minors = []
        for j in range(h):
            sub = [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]
            row_minors.append(determinant(sub))
        minor_matrix.append(row_minors)
    return minor_matrix

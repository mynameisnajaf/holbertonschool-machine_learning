#!/usr/bin/env python3
"""A module that calculates determinants and minor matrices."""


def determinant(matrix):
    """Calculate the determinant of a square matrix."""
    if not isinstance(matrix, list) or not \
all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    h = len(matrix)
    if h == 0:
        return 1
    if h == 1:
        return matrix[0][0]
    if h == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return (a * d) - (b * c)

    det = 0
    for col in range(h):
        minor_matrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        cofactor = ((-1) ** col) * matrix[0][col] * determinant(minor_matrix)
        det += cofactor
    return det


def minor(matrix):
    """Calculate the minor matrix of a square matrix."""
    if not isinstance(matrix, list) or not \
all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    h = len(matrix)
    if h == 0 or any(len(row) != h or len(row) == 0 for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = []
    for i in range(h):
        row_minors = []
        for j in range(h):
            sub_matrix = [
                row[:j] + row[j + 1:] for k, row in enumerate(matrix) if k != i
            ]
            row_minors.append(determinant(sub_matrix))
        minor_matrix.append(row_minors)
    return minor_matrix

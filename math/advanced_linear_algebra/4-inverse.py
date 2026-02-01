#!/usr/bin/env python3
"""A module that calculates determinants and adjugate matrices."""


def determinant(matrix):
    """Calculate the determinant of a square matrix."""
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
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


def cofactor(matrix):
    """Calculate the minor matrix of a square matrix."""
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")

    h = len(matrix)
    if h == 0 or any(len(row) != h or len(row) == 0 for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = []
    for i in range(h):
        row_cofactors = []
        for j in range(h):
            sub = [
                row[:j] + row[j + 1:] for k, row in enumerate(matrix) if k != i
            ]
            c = ((-1) ** (i + j)) * determinant(sub)
            row_cofactors.append(c)
        cofactor_matrix.append(row_cofactors)
    return cofactor_matrix


def adjugate(matrix):
    """Calculate the adjugate (adjoint) matrix."""
    cof = cofactor(matrix)

    adj = []
    h = len(cof)
    for j in range(h):
        row = []
        for i in range(h):
            row.append(cof[i][j])
        adj.append(row)

    return adj


def inverse(matrix):
    """Calculate the inverse of a square matrix."""
    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")
    h = len(matrix)
    if h == 0 or any(len(row) != h for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    h = len(adj)

    inv = [[adj[i][j] / det for j in range(h)] for i in range(h)]
    return inv

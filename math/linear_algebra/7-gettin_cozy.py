#!/usr/bin/env python3

"""A module that does the trick"""


def cat_matrices2D(mat1, mat2, axis=0):
    """A function that does the trick"""
    new_matrix =[]
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        new_matrix = [x[:] for x in mat1] + [x[:] for x in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            new_matrix.append(mat1[i][:] + mat2[i][:])

    else:
        return None
    return new_matrix

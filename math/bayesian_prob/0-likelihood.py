#!/usr/bin/env python3

"""A module that does the trick"""
import numpy as np


def likelihood(x, n, P):
    """A function that does the trick"""
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0'
        )
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    factorial = np.math.factorial
    numerator = factorial(n)
    denominator = factorial(x) * factorial(n - x)
    factor = numerator / denominator
    p_likelihood = factor * (np.power(P, x)) * (np.power((1 - P), (n - x)))
    return p_likelihood

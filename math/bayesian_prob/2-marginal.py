#!/usr/bin/env python3

"""A module that does the trick"""
import numpy as np


def marginal(x, n, P, Pr):
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
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError('Pr must sum to 1')
    factorial = np.math.factorial
    numerator = factorial(n)
    denominator = factorial(x) * factorial(n - x)
    factor = numerator / denominator
    p_likelihood = factor * (np.power(P, x)) * (np.power((1 - P), (n - x)))
    intersect = p_likelihood * Pr
    marg = 0
    for i in intersect:
        marg += i
    return marg

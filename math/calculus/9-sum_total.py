#!/usr/bin/env python3
"""A module that does the trick."""


def summation_i_squared(n):
    """Return the sum of squares from 1 to n."""
    if not isinstance(n, int) or n < 1:
        return None

    total = 0
    i = 1
    while i <= n:
        total += i ** 2
        i += 1
    return total

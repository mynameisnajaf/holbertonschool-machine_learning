#!/usr/bin/env python3

"""A module that does the trick"""


class Exponential:
    """A class that does the trick"""

    def __init__(self, data=None, lambtha=1.):
        """A function that does the trick"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = (1 / sum(data) / len(data))

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
                self.lambtha = (1 / (sum(data) / len(data)))

    def pdf(self, x):
        """Calculates the value of the PDF"""
        e = 2.7182818285
        if x < 0:
            return 0
        pdf = self.lambtha * (e**((-self.lambtha) * x)
        return pdf

    def cdf(self, x):
        """Calculates the value of the CDF"""
        e = 2.7182818285
        if x < 0:
            return 0
        cdf = 1 - (e**((-self.lambtha) * x)
        return cdf

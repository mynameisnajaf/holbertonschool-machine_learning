#!/usr/bin/env python3

"""A module that does the trick"""


class Poisson:
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
                self.lambtha = (sum(data) / len(data))

    def pmf(self, k):
        """A function Calculates the value of the PMF"""
        e = 2.7182818285
        k = int(k)
        factorial = 1
        if k < 0:
            return 0
        for i in range(1, k+1):
            factorial *= i
        pmf = e ** -self.lambtha * self.lambtha ** k / factorial
        return pmf

     def cdf(self, k):
         """A function Calculates the value of the CDF"""
         k = int(k)
         if k < 0:
             return 0
         cdf = 0
         for i in range(k+1):
             cdf += self.pmf(i)
         return cdf

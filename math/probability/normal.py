#!/usr/bin/env python3

"""A module that does the trick"""


class Normal:
    """
    Tye class to call methods of Normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """A function that does the trick"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = (sum(data) / len(data))
                summ = 0
                for i in data:
                    summ += (i - self.mean) ** 2

                self.stddev = (summ / len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value"""
        return z * self.stddev + self.mean

#!/usr/bin/env python3
"""A module that does the trick"""


class Normal:
    """The class to call methods of Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes the Normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
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

    def pdf(self, x):
        """Calculates the PDF"""
        pi = 3.1415926536
        e = 2.7182818285
        coef = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -(self.z_score(x) ** 2) / 2
        return coef * (e ** exponent)

    def cdf(self, x):
        """Calculates the CDF"""
        pi = 3.1415926536
        xa = self.z_score(x) / (2 ** 0.5)

        erf = (
            (2 / (pi ** 0.5)) *
            (
                xa -
                (xa ** 3) / 3 +
                (xa ** 5) / 10 -
                (xa ** 7) / 42 +
                (xa ** 9) / 216
            )
        )

        return (1 + erf) / 2

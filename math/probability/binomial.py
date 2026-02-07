#!/usr/bin/env python3

"""A module that does the trick"""


class Binomial:
    """The class to call methods of binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """init"""
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            summ = 0
            for i in data:
                summ += (i - mean) ** 2
            variance = (summ / len(data))
            self.p = 1 - (variance / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        n_fact = 1
        for i in range(1, self.n + 1):
            n_fact *= i
        k_fact = 1
        for i in range(1, k + 1):
            k_fact *= i
        nk_fact = 1
        for i in range(1, self.n - k + 1):
            nk_fact *= i
        comb = n_fact / (k_fact * nk_fact)
        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

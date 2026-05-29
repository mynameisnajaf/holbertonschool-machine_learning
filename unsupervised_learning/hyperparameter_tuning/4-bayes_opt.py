#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """A class that does the trick"""

    def __init__(self,
                 f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialise the class"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Acquisition function"""
        mu, sigma = self.gp.predict(self.X_s)

        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros_like(imp)
            Z[sigma > 0] = imp[sigma > 0] / sigma[sigma > 0]

            EI = np.zeros_like(imp)
            EI[sigma > 0] = (
                imp[sigma > 0] * norm.cdf(
                Z[sigma > 0]
            ) + sigma[sigma > 0] * norm.pdf(Z[sigma > 0])
            )

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

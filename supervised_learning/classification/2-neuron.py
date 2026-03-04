#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


class Neuron:
    """A class that does the trick"""

    def __init__(self, nx):
        """Initialize the class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return the weight matrix"""
        return self.__W

    @property
    def b(self):
        """Return the bias vector"""
        return self.__b

    @property
    def A(self):
        """Return the activation matrix"""
        return self.__A

    def forward_prop(self, X):
        """Forward propagation method"""
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return self.A

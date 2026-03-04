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
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

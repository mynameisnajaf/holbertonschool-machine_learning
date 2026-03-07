#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


class DeepNeuralNetwork:
    """A class that does the trick"""

    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for layer in layers:
            if not isinstance(layer, int) or layer <= 0:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for l in range(self.L):
            if l == 0:
                prev = nx
            else:
                prev = layers[l - 1]
            self.weights['W{}'.format(l + 1)] = (
                np.random.randn(layers[l], prev) * np.sqrt(2 / prev)
            )

            self.weights['b{}'.format(l + 1)] = np.zeros((layers[l], 1))

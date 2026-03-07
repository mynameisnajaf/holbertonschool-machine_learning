#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


class NeuralNetwork:
    """A class that represents a neural network"""

    def __init__(self, nx, nodes):
        """Constructor for the neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def w1_getter(self):
        """Get the weights of the first layer"""
        return self.__W1

    def b1_getter(self):
        """Get the biases of the first layer"""
        return self.__b1

    def w2_getter(self):
        """Get the weights of the second layer"""
        return self.__W2

    def b2_getter(self):
        """Get the biases of the second layer"""
        return self.__b2

    def a2_getter(self):
        """Get the activation function of the second layer"""
        return self.__A2

    def a1_getter(self):
        """Get the activation function of the first layer"""
        return self.__A1

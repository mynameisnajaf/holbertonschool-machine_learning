#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


class DeepNeuralNetwork:
    """A class that does the trick"""

    def __init__(self, nx, layers):
        """Constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for lay in range(self.__L):
            if layers[lay] < 1 or not isinstance(layers[lay], int):
                raise TypeError("layers must be a list of positive integers")

            if lay == 0:
                prev = nx
            else:
                prev = layers[lay - 1]
            self.__weights['W{}'.format(lay + 1)] = (
                np.random.randn(layers[lay], prev) * np.sqrt(2 / prev)
            )

            self.__weights['b{}'.format(lay + 1)] = np.zeros((layers[lay], 1))

    @property
    def weights(self):
        return self.__weights

    @property
    def cache(self):
        return self.__cache

    @property
    def L(self):
        return self.__L

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache['A0'] = X
        for lay in range(self.__L):
            W = self.__weights['W{}'.format(lay + 1)]
            b = self.__weights['b{}'.format(lay + 1)]
            A_prev = self.__cache['A{}'.format(lay)]

            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A{}'.format(lay + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Cost function"""
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-m_loss)
        return cost

    def evaluate(self, X, Y):
        """Evaluate the cost function"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent for deep neural network"""
        m = Y.shape[1]
        L = self.__L
        dz = cache[f'A{L}'] - Y

        for lay in reversed(range(L)):
            A_prev = cache[f'A{lay}']
            W = self.__weights[f'W{lay + 1}']

            dw = (1 / m) * np.matmul(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            self.__weights[f'W{lay + 1}'] -= alpha * dw
            self.__weights[f'b{lay + 1}'] -= alpha * db

            if lay > 0:
                dz = np.matmul(W.T, dz) * (A_prev * (1 - A_prev))
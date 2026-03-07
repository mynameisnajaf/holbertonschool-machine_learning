#!/usr/bin/env python3
"""Deep Neural Network for multiclass classification"""
import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Deep Neural Network supporting multiclass classification"""

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

            prev = nx if lay == 0 else layers[lay - 1]
            # He initialization
            self.__weights[f"W{lay + 1}"] = np.random.randn(layers[lay], prev) * np.sqrt(2 / prev)
            self.__weights[f"b{lay + 1}"] = np.zeros((layers[lay], 1))

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
            Z = np.matmul(self.__weights["W" + str(lay + 1)], self.__cache["A" + str(lay)])
            Z = Z + self.__weights["b" + str(lay + 1)]
            if lay == self.__L - 1:
                p1 = np.exp(Z)
                self.__cache["A" + str(lay + 1)] = p1 / np.sum(p1, axis=0, keepdims=True)  # softmax
            else:
                self.__cache["A" + str(lay + 1)] = 1 / (1 + np.exp(-Z))

        A_L = self.__cache[f"A{self.__L}"]
        return A_L, self.__cache

    def cost(self, Y, A):
        """Categorical cross-entropy cost"""
        m = Y.shape[1]
        C = (-1 / m) * np.sum(Y * np.log(A))
        return C

    def evaluate(self, X, Y):
        """Evaluate predictions for multiclass classification"""
        cache = self.__cache
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.argmax(A, axis=0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent (works for softmax last layer)"""
        m = Y.shape[1]
        for i in reversed(range(self.__L)):
            w = f"W{i + 1}"
            b = f"b{i + 1}"
            a = f"A{i + 1}"
            a_prev = f"A{i}"
            A = cache[a]
            A_prev = cache[a_prev]

            if i == self.__L - 1:
                dz = A - Y
                W = self.__weights[w]
            else:
                da = A * (1 - A)
                dz = np.dot(W.T, dz) * da
                W = self.__weights[w]

            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights[w] -= alpha * dw
            self.__weights[b] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            if i % step == 0 or i == iterations:
                costs.append(cost)
                steps.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Save model to file"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load model from file"""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

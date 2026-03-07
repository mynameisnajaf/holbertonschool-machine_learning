#!/usr/bin/env python3
"""Deep Neural Network (DNN) for multiclass classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """DNN class"""

    def __init__(self, nx, layers):
        """Init DNN with nx inputs and layer nodes"""
        if type(nx) is not int or nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a non-empty list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be positive integers")
            prev = nx if i == 0 else layers[i - 1]
            self.__weights[f"W{i+1}"] = np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            self.__weights[f"b{i+1}"] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Cache dict"""
        return self.__cache

    @property
    def weights(self):
        """Weights dict"""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache['A0'] = X
        for i in range(self.L):
            Z = np.matmul(self.__weights[f"W{i+1}"], self.__cache[f"A{i}"]) + self.__weights[f"b{i+1}"]
            if i == self.L - 1:
                exp_Z = np.exp(Z)
                self.__cache[f"A{i+1}"] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                self.__cache[f"A{i+1}"] = 1 / (1 + np.exp(-Z))
        return self.__cache[f"A{self.L}"], self.__cache

    def cost(self, Y, A):
        """Cross-entropy cost"""
        m = Y.shape[1]
        return (-1 / m) * np.sum(Y * np.log(A + 1e-8))

    def evaluate(self, X, Y):
        """Evaluate prediction"""
        A, _ = self.forward_prop(X)
        prediction = np.argmax(A, axis=0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One step gradient descent"""
        m = Y.shape[1]
        dZ_next = None
        for i in reversed(range(self.L)):
            A = cache[f"A{i+1}"]
            A_prev = cache[f"A{i}"]
            W = self.weights[f"W{i+1}"]
            if i == self.L - 1:
                dZ = A - Y
            else:
                dZ = np.dot(W_next.T, dZ_next) * (A * (1 - A))
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights[f"W{i+1}"] -= alpha * dW
            self.__weights[f"b{i+1}"] -= alpha * db
            dZ_next = dZ
            W_next = W

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Train network"""
        if type(iterations) is not int or iterations <= 0:
            raise ValueError("iterations must be positive int")
        if type(alpha) is not float or alpha <= 0:
            raise ValueError("alpha must be positive float")
        costs, steps = [], []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            if i % step == 0 or i == iterations:
                costs.append(cost)
                steps.append(i)
                if verbose: print(f"Cost after {i} iterations: {cost}")
            if i < iterations: self.gradient_descent(Y, cache, alpha)
        if graph:
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Save model"""
        if not filename.endswith(".pkl"): filename += ".pkl"
        with open(filename, "wb") as f: pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load model"""
        try:
            with open(filename, "rb") as f: return pickle.load(f)
        except FileNotFoundError:
            return None

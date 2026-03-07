#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np
import matplotlib.pyplot as plt


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

    def cost(self, Y, A):
        """Return the cost function"""
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-m_loss)
        return cost

    def evaluate(self, X, Y):
        """Evaluate the cost function"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Gradient descent method"""
        m = Y.shape[1]
        dz = (A - Y)
        dw = (1 / m) * (np.matmul(X, dz.transpose()).transpose())
        db = (1 / m) * np.sum(dz)
        self.__W = self.W - (alpha * dw)
        self.__b = self.b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):

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

        if graph:
            x_points = []
            y_points = []

        for i in range(iterations + 1):

            A = self.forward_prop(X)

            if i % step == 0:
                cost = self.cost(Y, A)

                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

                if graph:
                    x_points.append(i)
                    y_points.append(cost)

            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(x_points, y_points)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

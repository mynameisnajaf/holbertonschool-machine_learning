#!/usr/bin/env python3

"""A  module that does the trick"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """A function to plot"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    x = np.arange(0, 11)
    plt.plot(x, y, color="red")
    plt.xlim(0, 10)
    plt.show()

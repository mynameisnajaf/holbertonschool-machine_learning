#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """A function that does the trick"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    output_h = h - kh + 1
    output_w = w - kw + 1

    convolve = np.zeros((m, output_h, output_w))
    image = np.arange(m)

    for x in range(output_h):
        for y in range(output_w):
            convolve[image, x, y] = (np.sum(
                images[image, x:kh + x, y:kw + y] * kernel,
                axis=(1, 2)))
    return convolve

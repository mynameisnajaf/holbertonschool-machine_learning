#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """convolve_grayscale_padding"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]
    output_h = h + (2 * ph) - kh + 1
    output_w = w + (2 * pw) - kw + 1

    padding = np.pad(
        images,
        pad_width=((0, 0),
                   (ph, ph), (pw, pw)),
        mode='constant',
    )

    convolve = np.zeros((m, output_h, output_w))
    image = np.arange(m)

    for x in range(output_h):
        for y in range(output_w):
            convolve[image, x, y] = (np.sum(
                padding[image, x:kh + x, y:kw + y] * kernel,
                axis=(1, 2)))
    return convolve

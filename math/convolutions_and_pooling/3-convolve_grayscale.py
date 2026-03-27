#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """convolve the images using a kernel"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = 0
    pw = 0
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]

    padding = np.pad(images, pad_width=((0, 0), (ph, ph),
                                        (pw, pw)),
                     mode='constant')

    output_h = int(((h + 2 * ph - kh) / sh) + 1)
    output_w = int(((w + 2 * pw - kh) / sw) + 1)

    convolve = np.zeros((m, output_h, output_w))
    image = np.arange(m)

    for x in range(output_h):
        for y in range(output_w):
            convolve[image, x, y] = (np.sum(
                padding[image, x * sh:((x * sh) + kh),
                y * sw:((y * sw) + kw)] * kernel, axis=(1, 2)))
    return convolve

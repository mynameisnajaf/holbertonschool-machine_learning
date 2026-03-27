#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """convolve the images using a kernel"""
    m, h, w, c = images.shape
    kh, kw, kc = kernel_shape
    sh, sw = stride

    output_h = int(1 + ((h - kh) / sh))
    output_w = int(1 + ((w - kw) / sw))

    convolve = np.zeros((m, output_h, output_w, c))
    image = np.arange(m)

    for x in range(output_h):
        for y in range(output_w):
            if mode == 'max':
                convolve[image, x, y] = (np.max(images[image,
                x * sh:((x * sh) + kh),
                y * sw:((y * sw) + kw)],
                                                axis=(1, 2)))
            elif mode == 'avg':
                convolve[image, x, y] = (np.mean(images[image,
                x * sh:((x * sh) + kh),
                y * sw:((y * sw) + kw)],
                                                 axis=(1, 2)))
    return convolve

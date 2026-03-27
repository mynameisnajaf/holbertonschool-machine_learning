#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """convolve_grayscale_same"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    pad_h = int((kh - 1) / 2)
    pad_w = int((kw - 1) / 2)

    if kh % 2 == 0:
        pad_h = int(kh / 2)
    if kw % 2 == 0:
        pad_w = int(kw / 2)

    padding = np.pad(
        images,
        pad_width=((0, 0),
                    (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
    )

    convolve = np.zeros((m, h, w))
    image = np.arange(m)

    for x in range(h):
        for y in range(w):
            convolve[image, x, y] = (np.sum(padding[image,
                                            x:kh+x, y:kw+y] * kernel,
                                            axis=(1, 2)))
    return convolve

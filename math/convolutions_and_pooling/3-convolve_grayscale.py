#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """convolve the images using a kernel"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    ph, pw = 0, 0

    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    elif isinstance(padding, tuple):
        ph, pw = padding

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output_h = int(((h + 2 * ph - kh) / sh) + 1)
    output_w = int(((w + 2 * pw - kw) / sw) + 1)

    convolve = np.zeros((m, output_h, output_w))

    for x in range(output_h):
        for y in range(output_w):
            convolve[:, x, y] = np.sum(
                padded_images[
                    :, x * sh:(x * sh) + kh,
                    y * sw:(y * sw) + kw
                ] * kernel,
                axis=(1, 2)
            )

    return convolve

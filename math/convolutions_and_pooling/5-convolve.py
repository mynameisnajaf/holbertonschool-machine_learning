#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """convolve the images using a kernel"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    ph, pw = 0, 0

    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    elif isinstance(padding, tuple):
        ph, pw = padding

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    output_h = int(((h + 2 * ph - kh) / sh) + 1)
    output_w = int(((w + 2 * pw - kw) / sw) + 1)

    convolve = np.zeros((m, output_h, output_w, nc))
    image = np.arange(m)

    for x in range(output_h):
        for y in range(output_w):
            for z in range(nc):
                convolve[image, x, y, z] = np.sum(
                    padded_images[
                        :, x * sh:(x * sh) + kh,
                        y * sw:(y * sw) + kw
                    ] * kernels[:, :, :, z],
                    axis=(1, 2, 3)
                )

    return convolve

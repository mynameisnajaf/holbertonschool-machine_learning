#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Convolution with forward pass"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    c_h = int((h_prev - kh) / sh) + 1
    c_w = int((w_prev - kw) / sw) + 1

    conv = np.zeros((m, c_h, c_w, c_prev))

    for x in range(c_h):
        for y in range(c_w):
            region = A_prev[
                :,
                x * sh:(x * sh) + kh,
                y * sw:(y * sw) + kw,
                :
            ]

            if mode == 'max':
                conv[:, x, y, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                conv[:, x, y, :] = np.mean(region, axis=(1, 2))

    return conv

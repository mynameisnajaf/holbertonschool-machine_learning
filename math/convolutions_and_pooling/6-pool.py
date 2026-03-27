#!/usr/bin/env python3
"""Performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs max or average pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int(1 + (h - kh) / sh)
    output_w = int(1 + (w - kw) / sw)

    output = np.zeros((m, output_h, output_w, c))

    for x in range(output_h):
        for y in range(output_w):
            region = images[
                :,
                x * sh:(x * sh) + kh,
                y * sw:(y * sw) + kw,
                :
            ]

            if mode == 'max':
                output[:, x, y, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, x, y, :] = np.mean(region, axis=(1, 2))

    return output

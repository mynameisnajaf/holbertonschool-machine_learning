#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Convolution with forward pass"""
    m, h_prev ,w_prev ,c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    padh, padw = 0, 0
    if padding == "same":
        padh = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        padw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    if padding == "valid":
        padh, padw = 0, 0

    padding = np.pad(
        A_prev,
        pad_width=((0, 0),
                   (padh, padh), (padw, padw), (0, 0)),
        mode='constant',
    )

    c_h = int(((h_prev + 2 * padh - kh) / sh) + 1)
    c_w = int(((w_prev + 2 * padw - kw) / sw) + 1)

    conv = np.zeros((m, c_h, c_w, c_new))

    for i in range(c_h):
        for j in range(c_w):
            for k in range(c_new):
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw
                img_slice = padding[:, v_start:v_end, h_start:h_end, :]
                kernel = W[:, :, :, k]
                conv[:, i, j, k] = (np.sum(np.multiply(img_slice,
                                                       kernel),
                                           axis=(1, 2, 3)))
    Z = conv + b
    return activation(Z)

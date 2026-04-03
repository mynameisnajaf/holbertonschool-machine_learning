#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Convolve the backward function"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new= W.shape
    sh, sw = stride

    if padding == "same":
        padh = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        padw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    if padding == "valid":
        padh, padw = 0, 0

    A_padding = np.pad(
        A_prev,
        pad_width=((0, 0),
                   (padh, padh), (padw, padw), (0, 0)),
        mode='constant',
    )

    dA_padding = np.zeros_like(A_padding)
    dW_padding = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw

                slice_A = A_padding[:, v_start:v_end, h_start:h_end, :]

                dA_padding[:, v_start:v_end, h_start:h_end, :] += (
                        W[:, :, :, k] * dZ[:, i:i + 1, j:j + 1, k]
                )

                dW_padding[:, :, :, k] += np.sum(
                    slice_A * dZ[:, i:i + 1, j:j + 1, k],
                    axis=0
                )

    if padding == "same":
        dA_prev = dA_padding[:, padh:-padh, padw:-padw, :]
    else:
        dA_prev = dA_padding

    return dA_prev, dW_padding, db

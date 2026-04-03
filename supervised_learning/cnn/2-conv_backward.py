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

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_prev_pad = np.pad(A_prev, pad_width=((0, 0), (padh, padh), (padw, padw),
                                           (0, 0)), mode='constant')
    dA_prev_pad = np.pad(dA_prev, pad_width=((0, 0), (padh, padh), (padw, padw),
                                             (0, 0)), mode='constant')

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    a_slice = a_prev_pad[v_start:v_end, h_start:h_end]
                    da_prev_pad[v_start:v_end,
                    h_start:h_end] += \
                        W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if padding == 'same':
            dA_prev[i, :, :, :] += da_prev_pad[padh:-padh, padw:-padw, :]
        if padding == 'valid':
            dA_prev[i, :, :, :] += da_prev_pad

    return dA_prev, dW, db

#!/usr/bin/env python3
"""Performs back propagation over a convolutional layer"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs backpropagation over a convolutional layer"""

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                for c in range(c_new):

                    a_slice = A_prev_pad[i,
                                         vert_start:vert_end,
                                         horiz_start:horiz_end,
                                         :]

                    dA_prev_pad[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                :] += W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

    if padding == "same":
        if ph > 0 and pw > 0:
            dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
        elif ph > 0:
            dA_prev = dA_prev_pad[:, ph:-ph, :, :]
        elif pw > 0:
            dA_prev = dA_prev_pad[:, :, pw:-pw, :]
        else:
            dA_prev = dA_prev_pad
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db

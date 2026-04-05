#!/usr/bin/env python3
"""Performs back propagation over a convolutional layer"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs backpropagation over a convolutional layer"""

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # 1. Fixed Padding Calculation
    if padding == "same":
        # Use ceil to ensure the padded volume is large enough
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Pad the input and initialize gradient buffers
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
                # Calculate corners
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                for c in range(c_new):
                    # Slicing the padded input
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients
                    # The ValueError happens here if the slice is smaller than the filter
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

    # 2. Fixed Cropping Logic
    # Slicing from ph to ph + h_prev handles ph=0 correctly
    dA_prev = dA_prev_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]

    return dA_prev, dW, db

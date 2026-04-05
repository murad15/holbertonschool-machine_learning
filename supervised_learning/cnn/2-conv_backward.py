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
    dA_pd = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                # Calculate corners
                vs = h * sh
                ve = vs + kh
                hs = w * sw
                he = hs + kw

                for c in range(c_new):
                    # Slicing the padded input
                    a_slice = A_prev_pad[i, vs:ve, hs:he, :]

                    # Update gradients
                    # The ValueError happens here if slice smaller than filter
                    dA_pd[i, vs:ve, hs:he, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

    # 2. Fixed Cropping Logic
    # Slicing from ph to ph + h_prev handles ph=0 correctly
    dA_prev = dA_pd[:, ph:ph + h_prev, pw:pw + w_prev, :]

    return dA_prev, dW, db

#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Something that function does"""

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    oh = (h_prev + 2 * ph - kh) // sh + 1
    ow = (w_prev + 2 * pw - kw) // sw + 1

    output = np.zeros((m, oh, ow, c_new))

    for i in range(oh):
        for j in range(ow):
            img_slice = images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            for k in range(c_new):
                kernel = W[:, :, :, k]
                output[:, i, j, k] = np.sum(img_slice * kernel, axis=(1, 2, 3))
    Z = output + b
    result = activation(Z)

    return result

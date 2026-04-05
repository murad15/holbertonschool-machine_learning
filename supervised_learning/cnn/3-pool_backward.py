#!/usr/bin/env python3
"""Performs back propagation over a pooling layer"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs backpropagation over a pooling layer"""

    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape

    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):

                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                for ch in range(c):

                    if mode == "max":

                        a_slice = A_prev[i,
                                         vert_start:vert_end,
                                         horiz_start:horiz_end,
                                         ch]

                        mask = (a_slice == np.max(a_slice))

                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                ch] += mask * dA[i, h, w, ch]

                    elif mode == "avg":

                        da = dA[i, h, w, ch] / (kh * kw)

                        dA_prev[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                ch] += np.ones((kh, kw)) * da

    return dA_prev

#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Something that function does"""

    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1

    output = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            img_slice = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(img_slice, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(img_slice, axis=(1, 2))

    return output

#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Something that function does"""

    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant', constant_values=0)

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            for q in range(nc):
                img_slice = images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                output[:, i, j] = np.sum(img_slice * kernel[:,:,:,q], axis=(1, 2, 3))

    return output

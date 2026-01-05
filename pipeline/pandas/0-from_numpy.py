#!/usr/bin/env python3
"""This is project regarding pandas library"""


import pandas as pd

def from_numpy(array):
    # Ensure the array is 2D
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    n_cols = array.shape[1]
    columns = [chr(ord('A') + i) for i in range(n_cols)]

    return pd.DataFrame(array, columns=columns)

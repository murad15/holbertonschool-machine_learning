#!/usr/bin/env python3
"""This is project regarding pandas library"""


import pandas as pd
import numpy as np
import string

def from_numpy(array: np.ndarray) -> pd.DataFrame:
    # Ensure array is at least 2D for DataFrame construction
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    n_cols = array.shape[1]
    columns = list(string.ascii_uppercase[:n_cols])

    return pd.DataFrame(array, columns=columns)


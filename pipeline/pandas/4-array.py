#!/usr/bin/env python3
"""Some function to do something"""

import pandas as pd


def array(df):
    """This function does something interesting"""
    df1 = df.tail(10)
    df1 = df1.to_numpy()
    return df1

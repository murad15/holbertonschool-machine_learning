#!/usr/bin/env python3
"""Some function to do something"""


def array(df):
    """This function does something interesting"""
    return df.loc[:, ["High", "Close"]].tail(10).to_numpy()

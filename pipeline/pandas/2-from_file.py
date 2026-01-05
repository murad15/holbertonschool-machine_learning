#!/usr/bin/env python3
"""This is docstring for the project"""


import pandas as pd


def from_file(filename, delimiter):
    """Function to load data from file"""

    data = pd.read_csv(filename, delimeter = delimeter)
    return data

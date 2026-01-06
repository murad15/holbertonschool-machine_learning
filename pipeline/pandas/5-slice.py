#!/usr/bin/env python3
"""All your files should end with a new line"""


def slice(df):
    """All your files should end with a new line"""

    return df[["High", "Low", "Close", "Volume_BTC"]].iloc[::60]

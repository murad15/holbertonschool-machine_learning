#!/usr/bin/env python3
"""All your files should end with a new line"""


def slice(df):
    """All your files should end with a new line"""

    df = df[["High", "Low", "Close", "Volume_BTC"]]
    df = df.iloc[[i for i in range(0, len(df)+1, 60)
    return df

#!/usr/bin/env python3
"""asda asd ada sd dad asdwq e"""


def fill(df):
    """ADS asdads asa sd adasdasdas"""

    df = df.drop("Weighted_Price", axis=1)
    df["Close"] = df["Close"].ffill()
    for column in ["High", "Low", "Open"]:
        df[column] = df[column].fillna(df["Close"])
    for column in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[column] = df[column].fillna(0)
    return df

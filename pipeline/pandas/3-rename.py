#!/usr/bin/env python3
"""Something to write here and so on etc"""


def rename(df):
    """This function makes something interesting"""

    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[["Datetime", "Close"]]
    return df

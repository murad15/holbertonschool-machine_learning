#!/usr/bin/env python3
""" qweoqowe qewoqowe qoe oq eoqw oqow eroqeq qew"""


import pandas as pd


def hierarchy(df1, df2):
    """ooqowowq p p pqwpeqp epqpew pqwpepe"""

    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # Concatenate with keys
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # Rearrange MultiIndex so Timestamp is the first level
    df = df.swaplevel(0, 1).sort_index()

    return df

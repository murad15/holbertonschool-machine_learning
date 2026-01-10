#!/usr/bin/env python3
""" qweoqowe qewoqowe qoe oq eoqw oqow eroqeq qew"""


import pandas as pd


def hierarchy(df1, df2):
    """ooqowowq p p pqwpeqp epqpew pqwpepe"""

    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)
    df1.swaplevel()
    df2.swaplevel()
    df1 = df1.loc[1417411980 <= df1.index <= 1417417980]
    df2 = df2.loc[1417411980 <= df2.index <= 1417417980]
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    return df.sort_index()

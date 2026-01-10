#!/usr/bin/env python3
"""kasda aslkdnald asq adnasdas kljdnalkns asr qqwerw"""


import pandas as pd
def concat(df1, df2):
    """ asdasd asd aas dasd asd a dgfwsf 3ewwer """


    # Import the index function
    index = __import__('10-index').index

    # Index both dataframes on the Timestamp column
    df1 = index(df1)
    df2 = index(df2)

    # Select rows from df2 up to and including the given timestamp
    df2 = df2.loc[df2.index <= 1417411920]

    # Concatenate df2 on top of df1 with keys
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

#!/usr/bin/env python3
"""clean up the raw btc data and save it ready for training"""
import sys
import numpy as np
import pandas as pd


# columns we keep, in this order. Close ends up at index 3
FEATURES = ['Open', 'High', 'Low', 'Close',
            'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']


def load_hourly(path):
    """read one csv, drop the junk and resample to 1 hour"""
    df = pd.read_csv(path)
    df = df.dropna()

    # unix seconds -> real timestamps so we can resample
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.set_index('Timestamp')

    # the early years are basically dead, no point feeding them in
    df = df[df.index >= '2017-01-01']

    # 1 min rows -> 1 hour rows. each column gets aggregated its own way
    h = pd.DataFrame()
    h['Open'] = df['Open'].resample('1h').first()
    h['High'] = df['High'].resample('1h').max()
    h['Low'] = df['Low'].resample('1h').min()
    h['Close'] = df['Close'].resample('1h').last()
    h['Volume_(BTC)'] = df['Volume_(BTC)'].resample('1h').sum()
    h['Volume_(Currency)'] = df['Volume_(Currency)'].resample('1h').sum()
    h['Weighted_Price'] = df['Weighted_Price'].resample('1h').mean()

    # resampling over gaps leaves NaNs, drop them
    h = h.dropna()
    return h[FEATURES]


def main():
    # bitstamp covers the longest span, use it as the main source
    path = sys.argv[1] if len(sys.argv) > 1 else 'bitstamp.csv'
    data = load_hourly(path).values.astype('float32')

    # chronological split, no shuffling for time series
    n = len(data)
    cut = int(n * 0.8)
    train, val = data[:cut], data[cut:]

    # scale with train stats only so val stays honest
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    train = (train - mean) / std
    val = (val - mean) / std

    np.savez('btc_data.npz', train=train, val=val, mean=mean, std=std)
    print('saved {} train rows and {} val rows'.format(len(train), len(val)))


if __name__ == '__main__':
    main()

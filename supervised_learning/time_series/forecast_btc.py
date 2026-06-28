#!/usr/bin/env python3
"""build, train and validate an LSTM to forecast the next hour BTC close"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


SEQ_LEN = 24          # use the past 24 hours
CLOSE_IDX = 3         # Close column position from preprocess_data.py


def windowed(data, batch=64, shuffle=False):
    """slide a 25-row window over data: 24 hours in, next close out"""
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(SEQ_LEN + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(SEQ_LEN + 1))
    # first 24 rows are the input, close of the 25th is the target
    ds = ds.map(lambda w: (w[:-1], w[-1, CLOSE_IDX]))
    if shuffle:
        ds = ds.shuffle(1000)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(n_features):
    """small two layer LSTM, regression head"""
    model = keras.Sequential([
        keras.layers.Input((SEQ_LEN, n_features)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main():
    d = np.load('btc_data.npz')
    train, val = d['train'], d['val']

    train_ds = windowed(train, shuffle=True)
    val_ds = windowed(val)

    model = build_model(train.shape[1])
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=20)
    model.save('btc_forecast.keras')


if __name__ == '__main__':
    main()

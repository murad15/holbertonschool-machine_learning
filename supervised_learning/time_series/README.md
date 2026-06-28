# Bitcoin Forecasting with an RNN

Predicting the BTC close price one hour ahead from the previous 24 hours of
data, using an LSTM trained on the Bitstamp/Coinbase minute datasets.

## Files

- `preprocess_data.py` — cleans the raw csv and writes `btc_data.npz`
- `forecast_btc.py` — builds, trains and validates the model
- `README.md` — this file

## Preprocessing choices

A few things about the raw data drove the decisions here:

- **Not all rows are useful.** The raw files have plenty of `NaN` rows from
  minutes with no trades, so those get dropped.
- **The old data isn't worth much.** Bitstamp goes back to 2012 when volume
  was tiny and prices barely moved, so I only keep data from 2017 on.
- **The minute resolution is too fine.** We only care about the next hour, so
  the 1-minute rows are resampled to hourly windows (open = first, high = max,
  low = min, close = last, volumes summed, weighted price averaged).
- **Scaling matters.** Prices and volumes live on very different ranges, so
  everything is standardized. The mean/std are computed on the training split
  only and reused on validation, so no future info leaks in.
- The cleaned arrays plus the scaling stats are saved to a single `.npz`.

## Model

The input is a window of the last 24 hourly steps (all 7 features) and the
target is the close price of the following hour. Data is fed through a
`tf.data.Dataset` that slides a 25-row window across the series. The network is
a two-layer LSTM with dropout and a single dense output, trained with Adam and
MSE loss.

## Usage

```
python preprocess_data.py bitstamp.csv
python forecast_btc.py
```

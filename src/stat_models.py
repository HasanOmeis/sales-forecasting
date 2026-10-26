import warnings  # lets us hide warning messages
import numpy as np  # math on arrays
import pandas as pd  # tables and time series
from statsmodels.tsa.statespace.sarimax import SARIMAX  # the SARIMA model from statsmodels

YEAR = 52.18  # average number of weeks in a year (365.25 / 7)


def fourier_terms(start, length, n_terms=2):  # build sin/cos waves that describe the yearly cycle
    t = np.arange(start, start + length)  # week positions, e.g. 0..168 for training or 169..176 for forecasting
    cols = {}  # will hold each wave as a column
    for k in range(1, n_terms + 1):  # one pair of waves per term (k=1 is one cycle a year, k=2 is two cycles)
        cols[f"sin_{k}"] = np.sin(2 * np.pi * k * t / YEAR)  # sine wave with k cycles per year
        cols[f"cos_{k}"] = np.cos(2 * np.pi * k * t / YEAR)  # cosine wave with k cycles per year
    return pd.DataFrame(cols)  # table with one row per week and one column per wave


def sarima_forecast(train, horizon, order=(1, 1, 1), n_terms=2):  # forecast one series with SARIMA + Fourier terms
    y = np.asarray(train, dtype=float)  # the training sales as a plain array
    exog_train = fourier_terms(0, len(y), n_terms)  # yearly waves for the training weeks
    exog_future = fourier_terms(len(y), horizon, n_terms)  # the same waves continued into the forecast weeks
    model = SARIMAX(y, exog=exog_train, order=order)  # ARIMA(1,1,1): 1 past value, 1 difference (removes trend), 1 past error
    with warnings.catch_warnings():  # open a block where warnings are hidden
        warnings.simplefilter("ignore")  # hide convergence warnings (common on short, noisy series)
        fit = model.fit(disp=False)  # fit the model without printing progress
    pred = fit.forecast(steps=horizon, exog=exog_future)  # forecast the next weeks using the future waves
    return np.clip(np.asarray(pred, dtype=float), 0, None)  # drop the number index (plain array), then turn negatives into 0


def sarima_all(weekly_train, horizon):  # forecast every column of a wide table (same shape as the LightGBM functions)
    preds = {}  # will hold each category's forecast
    for cat in weekly_train.columns:  # loop over the sub-categories in the table
        preds[cat] = sarima_forecast(weekly_train[cat], horizon)  # forecast this category on its own
    future_index = pd.date_range(weekly_train.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W")  # dates of the forecast weeks
    return pd.DataFrame(preds, index=future_index)  # wide table: one row per forecast week, one column per category
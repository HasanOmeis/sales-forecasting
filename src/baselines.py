import numpy as np  # numpy lets us build arrays of repeated forecast values


def naive(train, horizon):  # naive baseline: forecast the last seen week for every future week
    train = np.asarray(train, dtype=float)  # turn the training history into a numpy array
    return np.repeat(train[-1], horizon)  # repeat the final training value once per forecast week


def seasonal_naive(train, horizon, season=52):  # seasonal naive: forecast the same week from last year
    train = np.asarray(train, dtype=float)  # turn the training history into a numpy array
    last_season = train[-season:]  # take the most recent 52 weeks, which is one full year of history
    reps = int(np.ceil(horizon / season))  # how many times we need to repeat that year to cover the horizon (1 for 8 weeks)
    return np.tile(last_season, reps)[:horizon]  # repeat the year if needed, then keep only the first 8 values


def moving_average(train, horizon, window=4):  # moving average baseline: forecast the mean of the last 4 weeks
    train = np.asarray(train, dtype=float)  # turn the training history into a numpy array
    return np.repeat(train[-window:].mean(), horizon)  # average the last 4 weeks and repeat that value for every forecast week
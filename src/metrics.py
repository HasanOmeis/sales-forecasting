import numpy as np  # numpy gives us fast math on arrays of numbers


def mae(actual, forecast):  # Mean Absolute Error: the average size of the miss, in sales dollars
    actual = np.asarray(actual, dtype=float)  # turn the real values into a numpy array so we can do math on them
    forecast = np.asarray(forecast, dtype=float)  # same for the forecast values
    return np.mean(np.abs(actual - forecast))  # take each miss, drop its sign, then average them


def rmse(actual, forecast):  # Root Mean Squared Error: like MAE but punishes big misses more
    actual = np.asarray(actual, dtype=float)  # real values as an array
    forecast = np.asarray(forecast, dtype=float)  # forecast values as an array
    return np.sqrt(np.mean((actual - forecast) ** 2))  # square the misses, average them, then undo the square with a root


def smape(actual, forecast):  # symmetric MAPE: the error as a percentage (0 to 200), so series of different sizes can be compared
    actual = np.asarray(actual, dtype=float)  # real values as an array
    forecast = np.asarray(forecast, dtype=float)  # forecast values as an array
    denom = (np.abs(actual) + np.abs(forecast)) / 2  # the average size of actual and forecast, used to scale each miss
    diff = np.abs(actual - forecast)  # the absolute miss for each week
    ratio = np.where(denom == 0, 0.0, diff / np.where(denom == 0, 1.0, denom))  # divide safely: if both are zero, count the miss as 0 instead of crashing
    return 100 * np.mean(ratio)  # average the ratios and express them as a percentage


def mase(actual, forecast, train):  # Mean Absolute Scaled Error: our error divided by the error of a simple "last week" guess on the training data
    actual = np.asarray(actual, dtype=float)  # real values as an array
    forecast = np.asarray(forecast, dtype=float)  # forecast values as an array
    train = np.asarray(train, dtype=float)  # the training history, used to measure how hard the series is
    scale = np.mean(np.abs(np.diff(train)))  # the average week-to-week change in training: the error a naive forecast would make
    if scale == 0:  # a perfectly flat history would make us divide by zero
        return np.nan  # return "not a number" instead of crashing
    return np.mean(np.abs(actual - forecast)) / scale  # below 1 means we beat the naive guess, above 1 means we are worse
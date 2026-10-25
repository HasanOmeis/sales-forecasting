import numpy as np  # numpy for arrays and clipping
import pandas as pd  # pandas for dates and tables
import lightgbm as lgb  # the LightGBM library
from src.features import build_features, FEATURES  # our feature pipeline and the list of model inputs

DEFAULT_PARAMS = {  # starting settings for LightGBM (we tune these on Day 5)
    "n_estimators": 300,  # number of trees to build
    "learning_rate": 0.05,  # how much each tree corrects the previous ones (small = careful)
    "num_leaves": 15,  # max leaves per tree (kept small because our dataset is small)
    "min_child_samples": 20,  # each leaf needs at least 20 rows, which limits overfitting
    "random_state": 42,  # fixed seed so results are repeatable
    "verbose": -1,  # hide LightGBM's training messages
}


def train_lgbm(weekly_train, params=None):  # train one LightGBM model on a wide weekly table
    params = params or DEFAULT_PARAMS  # use the default settings if none are given
    feat = build_features(weekly_train)  # turn the wide table into the feature table
    feat = feat.dropna(subset=["sales"])  # keep only rows with a real target value
    model = lgb.LGBMRegressor(**params)  # create the model with our settings
    model.fit(feat[FEATURES], feat["sales"])  # learn to predict sales from the features (NaN features are handled by LightGBM)
    return model  # give back the trained model


def recursive_forecast(model, weekly_train, horizon=8):  # forecast the next weeks one at a time
    history = weekly_train.copy()  # copy of the known weeks, so we can add predicted rows to it
    for _ in range(horizon):  # repeat once per week we need to forecast
        next_date = history.index[-1] + pd.Timedelta(weeks=1)  # the date of the next week
        history.loc[next_date] = np.nan  # add an empty row for that week (its sales are unknown)
        feat = build_features(history)  # rebuild features, so lags now include earlier predictions
        rows = feat[feat["date"] == next_date]  # keep only the rows for the week we are predicting
        preds = np.clip(model.predict(rows[FEATURES]), 0, None)  # predict, and turn negatives into 0 (sales can't be negative)
        preds = pd.Series(preds, index=rows["category"].astype(str).values)  # label each prediction with its category name
        history.loc[next_date] = preds[history.columns].values  # write predictions into the row in the right column order
    return history.iloc[-horizon:]  # return only the forecast weeks

def global_forecast(weekly_train, horizon):  # one model for all categories: train it, then forecast
    model = train_lgbm(weekly_train)  # train on all 9 categories together
    return recursive_forecast(model, weekly_train, horizon)  # forecast every category for the horizon

def per_category_forecast(weekly_train, horizon):  # one separate model per category
    preds = {}  # will hold each category's forecast
    for cat in weekly_train.columns:  # loop over the 9 sub-categories
        one = weekly_train[[cat]]  # a wide table with only this category's column
        model = train_lgbm(one)  # train a model on this category's history only
        preds[cat] = recursive_forecast(model, one, horizon)[cat]  # forecast this category for the horizon
    return pd.DataFrame(preds)  # combine the 9 forecasts into one wide table
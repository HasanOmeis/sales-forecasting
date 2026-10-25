import pandas as pd  # pandas for building the results table
from src.validation import walk_forward_splits  # Day 2 fold function (same folds as the baselines)
from src.metrics import mae, rmse, smape, mase  # Day 2 metrics


def evaluate(weekly, forecast_fn, model_name, n_folds=5, horizon=8):  # score any forecasting method on the walk-forward folds
    rows = []  # one result row per (fold, category)
    splits = walk_forward_splits(len(weekly), n_folds, horizon)  # the same 5 folds used on Day 2
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):  # loop over folds, numbered from 1
        train = weekly.iloc[train_idx]  # known weeks up to the cutoff
        test = weekly.iloc[test_idx]  # the real sales we try to forecast
        pred = forecast_fn(train, len(test_idx))  # the method forecasts all categories for the horizon
        for cat in weekly.columns:  # score each category separately
            rows.append({  # store this fold and category's scores
                "model": model_name,  # which method produced the forecast
                "fold": fold,  # fold number 1 to 5
                "category": cat,  # sub-category name
                "mae": mae(test[cat], pred[cat]),  # average miss in dollars
                "rmse": rmse(test[cat], pred[cat]),  # miss with big errors punished more
                "smape": smape(test[cat], pred[cat]),  # miss as a percentage
                "mase": mase(test[cat], pred[cat], train[cat]),  # miss compared to a naive guess
            })
    return pd.DataFrame(rows)  # results table: one row per fold and category
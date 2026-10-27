import pandas as pd  # to compare tables and series
import pytest  # the testing framework

from src.data import load_raw, make_weekly_table  # build the weekly table
from src.features import build_features  # your Day 3 feature function

LAGS = [1, 2, 4, 8, 52]  # the lag weeks you built
WINDOWS = [4, 8, 12]  # the rolling window sizes you built


@pytest.fixture(scope="module")  # build once, share across all tests in this file
def weekly():  # the weekly sales table (dates x sub-categories)
    return make_weekly_table(load_raw("data/superstore.csv"))  # same pipeline as the API uses


@pytest.fixture(scope="module")  # build once, share across all tests in this file
def feats(weekly):  # the feature table built from the weekly table
    df = build_features(weekly)  # long format: date, category, sales, features
    return df.sort_values(["category", "date"]).reset_index(drop=True)  # order by category then time so shifts line up


def feature_columns(df):  # helper: every column that is a model input
    return [c for c in df.columns if c not in ("date", "category", "sales")]  # everything except the ids and the target


@pytest.mark.parametrize("k", LAGS)  # run this test once for each lag
def test_lag_is_exactly_k_weeks_back(feats, k):  # lag_k must equal sales from k weeks earlier, same category
    expected = feats.groupby("category")["sales"].shift(k)  # sales moved down k weeks inside each category
    pd.testing.assert_series_equal(feats[f"lag_{k}"], expected, check_names=False)  # must match row for row


@pytest.mark.parametrize("w", WINDOWS)  # run this test once for each window size
def test_rolling_mean_uses_only_past(feats, w):  # roll_mean_w must be built from previous weeks only
    expected = feats.groupby("category")["sales"].transform(  # compute the correct value per category
        lambda s: s.shift(1).rolling(w, min_periods=1).mean()  # shift first so this week is excluded, then average
    )
    has_value = feats[f"roll_mean_{w}"].notna()  # only compare rows where your feature has a value
    pd.testing.assert_series_equal(  # the two must match on those rows
        feats.loc[has_value, f"roll_mean_{w}"], expected[has_value], check_names=False  # your feature vs expected
    )


@pytest.mark.parametrize("w", WINDOWS)  # run this test once for each window size
def test_rolling_std_uses_only_past(feats, w):  # roll_std_w must be built from previous weeks only
    expected = feats.groupby("category")["sales"].transform(  # compute the correct value per category
        lambda s: s.shift(1).rolling(w, min_periods=1).std()  # shift first so this week is excluded, then std
    )
    has_value = feats[f"roll_std_{w}"].notna()  # only compare rows where your feature has a value
    pd.testing.assert_series_equal(  # the two must match on those rows
        feats.loc[has_value, f"roll_std_{w}"], expected[has_value], check_names=False  # your feature vs expected
    )


def test_changing_the_future_does_not_change_the_past(weekly, feats):  # the strongest leakage test
    cutoff = weekly.index[-10]  # pick a date 10 weeks before the end
    future_changed = weekly.copy()  # copy so the real table isn't touched
    future_changed.loc[future_changed.index >= cutoff] = 999999.0  # replace every sale from the cutoff onward with a huge number
    new = build_features(future_changed)  # rebuild features on the altered data
    new = new.sort_values(["category", "date"]).reset_index(drop=True)  # same ordering as the original
    cols = feature_columns(feats)  # compare only the model inputs, not the target
    keep_old = feats["date"] <= cutoff  # rows up to and including the cutoff week
    keep_new = new["date"] <= cutoff  # same rows in the rebuilt table
    pd.testing.assert_frame_equal(  # features for these weeks must be identical
        feats.loc[keep_old, cols].reset_index(drop=True),  # original features
        new.loc[keep_new, cols].reset_index(drop=True),  # features after changing the future
    )
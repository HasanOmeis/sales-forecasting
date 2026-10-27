import pandas as pd  # to work with dates and the weekly table
import pytest  # the testing framework

from src.data import TOP_SUBCATEGORIES, load_raw, make_weekly_table  # your Day 1 data functions and the 9-category list


@pytest.fixture(scope="module")  # build the table once and share it across all tests in this file
def weekly():  # the fixture's name; tests ask for it by putting "weekly" in their arguments
    raw = load_raw("data/superstore.csv")  # read the raw Superstore CSV
    return make_weekly_table(raw)  # turn it into the weekly sales table (weeks x sub-categories)


def test_shape(weekly):  # check the table has the size we expect
    assert weekly.shape == (209, 9)  # 209 weeks and 9 sub-categories, as found on Day 1


def test_columns_are_top_subcategories(weekly):  # check the columns are exactly the 9 chosen sub-categories
    assert set(weekly.columns) == set(TOP_SUBCATEGORIES)  # same names, order doesn't matter


def test_no_missing_weeks(weekly):  # the main test: no week is skipped
    dates = pd.to_datetime(weekly.index)  # make sure the index is real dates
    gaps = dates.to_series().diff().dropna()  # time between each week and the one before it
    assert (gaps == pd.Timedelta(days=7)).all()  # every step must be exactly 7 days, so no gaps and no duplicates


def test_dates_sorted_and_unique(weekly):  # weeks must be in order with no repeats
    assert weekly.index.is_monotonic_increasing  # dates go from oldest to newest
    assert weekly.index.is_unique  # no week appears twice


def test_no_missing_values(weekly):  # weeks with no sales should be 0, not empty
    assert not weekly.isna().any().any()  # no NaN anywhere in the table


def test_no_negative_sales(weekly):  # sales can't be below zero
    assert (weekly >= 0).all().all()  # every value is 0 or more
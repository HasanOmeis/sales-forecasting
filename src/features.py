import pandas as pd  # pandas for working with tables


def to_long(weekly):  # turn the wide weekly table into one row per (week, category)
    long = weekly.reset_index()  # move the date index into a normal column
    long = long.rename(columns={long.columns[0]: "date"})  # name that first column "date"
    long = long.melt(id_vars="date", var_name="category", value_name="sales")  # stack the 9 category columns into rows
    long = long.sort_values(["category", "date"]).reset_index(drop=True)  # sort by category then date so each series is in time order
    return long  # give back the long table

def add_lags(long, lags=(1, 2, 4, 8, 52)):  # add past sales values as new columns
    long = long.copy()  # work on a copy so the original table is not changed
    grouped = long.groupby("category")["sales"]  # group sales by category so each series is handled separately
    for lag in lags:  # loop over each lag size
        long[f"lag_{lag}"] = grouped.shift(lag)  # shift sales down by "lag" weeks within each category
    return long  # give back the table with the new lag columns

def add_rolling(long, windows=(4, 8, 12)):  # add rolling mean and std of past sales
    long = long.copy()  # work on a copy so the original table is not changed
    grouped = long.groupby("category")["sales"]  # group sales by category so each series is handled separately
    for w in windows:  # loop over each window size
        long[f"roll_mean_{w}"] = grouped.transform(lambda s: s.shift(1).rolling(w).mean())  # mean of the previous w weeks (shift 1 = exclude this week)
        long[f"roll_std_{w}"] = grouped.transform(lambda s: s.shift(1).rolling(w).std())  # spread of the previous w weeks (shift 1 = exclude this week)
    return long  # give back the table with the new rolling columns

def add_calendar(long):  # add features describing when each week is
    long = long.copy()  # work on a copy so the original table is not changed
    long["week_of_year"] = long["date"].dt.isocalendar().week.astype(int)  # week number 1 to 53
    long["month"] = long["date"].dt.month  # month number 1 to 12
    long["is_q4"] = (long["month"] >= 10).astype(int)  # 1 for Oct, Nov, Dec (holiday season), else 0
    return long  # give back the table with the new calendar columns

def build_features(weekly):  # full pipeline: wide weekly table in, model-ready table out
    long = to_long(weekly)  # wide to long: one row per (week, category)
    long = add_lags(long)  # add lag_1, lag_2, lag_4, lag_8, lag_52
    long = add_rolling(long)  # add rolling mean and std (shifted, no leakage)
    long = add_calendar(long)  # add week_of_year, month, is_q4
    long["category"] = long["category"].astype("category")  # mark category as categorical so LightGBM treats it as a label, not text
    return long  # give back the finished feature table


FEATURES = [  # list of columns the model is allowed to use (sales is the target, date is not a feature)
    "category",  # which sub-category the row belongs to
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_52",  # past sales values
    "roll_mean_4", "roll_std_4", "roll_mean_8", "roll_std_8", "roll_mean_12", "roll_std_12",  # recent averages and spreads
    "week_of_year", "month", "is_q4",  # calendar features
]


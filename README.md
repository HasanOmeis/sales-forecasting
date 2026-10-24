# Multi-Category Sales Forecasting

## Problem
Forecast weekly sales for each product sub-category of a multi-category retail store, and build the solution as a reproducible ML pipeline.

## Dataset
Superstore Sales Dataset (Kaggle, by rohitsahoo): 9,800 order lines, January 2015 to December 2018. Only `Sales` is numeric (no discount, quantity, or profit). The data is not included in this repo: download `train.csv` from Kaggle and save it as `data/superstore.csv`.

## Data preparation
- Converted `Order Date` to a real date (day/month/year format).
- Removed 1 duplicated row (identical in every column except `Row ID`).
- Aggregated sales to weekly totals per sub-category, filling empty weeks with 0.
- Modeling 9 sub-categories with few zero-sales weeks: Phones, Chairs, Storage, Binders, Accessories, Appliances, Furnishings, Paper, Tables.
- Left out Copiers and Machines (large sales but mostly zero weeks) and small sub-categories such as Fasteners, Labels, Envelopes, and Supplies.

## Key findings from EDA
- Sales peak in September, November, and December.
- There is a repeating yearly cycle.
- There is an upward trend over the four years.

## Project status
Day 1 done (data preparation and EDA). Baselines, models, API, and Docker are coming.

## Validation and baselines

**Validation:** walk-forward with an expanding window. 5 folds, each forecasting the next 8 weeks. The last 40 weeks of the 209-week series are used for testing, and the first training window is 169 weeks. The final fold covers the Q4 peak.

**Metrics:** MAE, RMSE, sMAPE and MASE, implemented in `src/metrics.py`.

**Baselines** (in `src/baselines.py`): naive (last week), seasonal naive (same week last year) and a 4-week moving average.

**Baseline results** (average over 5 folds and 9 sub-categories):

| Model | MAE | RMSE | sMAPE | MASE |
|---|---|---|---|---|
| 4-week moving average | 987 | 1206 | 82.9 | 1.23 |
| Seasonal naive | 1127 | 1460 | 98.5 | 1.38 |
| Naive | 1133 | 1385 | 91.1 | 1.45 |

The 4-week moving average is the best baseline, so it is the bar every later model must beat. Seasonal naive is no better than naive because the series are short and noisy.
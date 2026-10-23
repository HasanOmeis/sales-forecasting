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
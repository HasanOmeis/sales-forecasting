import pandas as pd  # to combine history and forecast into one table
import streamlit as st  # the dashboard library

from src.predict import MAX_HORIZON, forecast, load_model  # your Day 5 model loader and forecast function

st.set_page_config(page_title="Sales Forecast", layout="wide")  # browser tab title and use the full page width
st.title("Weekly Sales Forecast by Sub-Category")  # big heading at the top
st.caption("Global LightGBM model trained on Superstore weekly sales (2015–2018)")  # one-line description under the title

bundle = load_model()  # load the saved model bundle once (cached)
history = bundle["history"]  # the weekly sales table: dates x 9 sub-categories

col1, col2 = st.columns(2)  # two side-by-side controls
sub_category = col1.selectbox("Sub-category", sorted(bundle["categories"]))  # dropdown with the 9 sub-categories
horizon = col2.slider("Weeks ahead", min_value=1, max_value=MAX_HORIZON, value=8)  # slider for the horizon, default 8

points = forecast(sub_category, horizon)  # get the forecast as a list of {"week", "forecast"} dicts
fc = pd.Series(  # turn it into a date-indexed series
    [p["forecast"] for p in points],  # the predicted values
    index=pd.to_datetime([p["week"] for p in points]),  # the forecast dates as real dates
    name="Forecast",  # line name in the chart legend
)  # closes pd.Series(
actual = history[sub_category].iloc[-104:].rename("Actual")  # the last 2 years of real sales for this sub-category
chart_data = pd.concat([actual, fc], axis=1)  # one table with an Actual and a Forecast column (empty where a line has no data)

st.line_chart(chart_data)  # draw both lines on one chart

m1, m2, m3 = st.columns(3)  # three small summary numbers side by side
m1.metric("Forecast total", f"{fc.sum():,.0f}")  # sum of predicted sales over the horizon
m2.metric("Avg per week (forecast)", f"{fc.mean():,.0f}")  # average predicted weekly sales
m3.metric("Avg per week (last 8 actual)", f"{actual.iloc[-8:].mean():,.0f}")  # recent real average, for comparison

st.subheader("Forecast table")  # heading above the table
st.dataframe(pd.DataFrame(points), hide_index=True)  # the raw forecast rows, the same as the API returns
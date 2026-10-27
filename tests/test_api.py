import pandas as pd  # to check the forecast dates
import pytest  # the testing framework
from fastapi.testclient import TestClient  # calls the API in memory, no server needed

from src.api import app  # your FastAPI app

client = TestClient(app)  # a fake browser that sends requests straight to the app


def test_health():  # the API is alive
    response = client.get("/health")  # call /health
    assert response.status_code == 200  # 200 means success
    assert response.json() == {"status": "ok"}  # exact reply we expect


def test_categories_lists_nine():  # /categories returns the 9 trained sub-categories
    response = client.get("/categories")  # call /categories
    assert response.status_code == 200  # success
    assert len(response.json()["categories"]) == 9  # exactly 9 names


@pytest.mark.parametrize("horizon", [1, 8, 26])  # try the smallest, the default, and the largest horizon
def test_forecast_shape(horizon):  # the reply has the right structure and length
    response = client.get("/forecast", params={"sub_category": "Phones", "horizon": horizon})  # ask for a forecast
    assert response.status_code == 200  # success
    body = response.json()  # turn the reply into a Python dict
    assert body["sub_category"] == "Phones"  # echoes the sub-category
    assert body["horizon"] == horizon  # echoes the horizon
    assert len(body["forecast"]) == horizon  # exactly one prediction per week asked for
    for point in body["forecast"]:  # check every row
        assert set(point) == {"week", "forecast"}  # only these two keys
        assert isinstance(point["forecast"], float)  # the prediction is a number


def test_forecast_weeks_are_consecutive():  # weeks start right after the data and step by 7 days
    response = client.get("/forecast", params={"sub_category": "Chairs", "horizon": 8})  # ask for 8 weeks of Chairs
    weeks = pd.to_datetime([p["week"] for p in response.json()["forecast"]])  # pull out the dates
    assert weeks[0] == pd.Timestamp("2019-01-06")  # first week is the one after 2018-12-30
    assert (weeks.to_series().diff().dropna() == pd.Timedelta(days=7)).all()  # every step is exactly one week


def test_unknown_category_returns_400():  # a sub-category the model never saw
    response = client.get("/forecast", params={"sub_category": "Pizza", "horizon": 8})  # ask for Pizza
    assert response.status_code == 400  # rejected as a bad request


@pytest.mark.parametrize("horizon", [0, 27])  # one value below the limit, one above
def test_bad_horizon_returns_422(horizon):  # horizon must be between 1 and 26
    response = client.get("/forecast", params={"sub_category": "Phones", "horizon": horizon})  # ask with a bad horizon
    assert response.status_code == 422  # FastAPI's validation error
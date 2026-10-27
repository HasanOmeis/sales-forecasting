from fastapi import FastAPI, HTTPException, Query  # app class, error responses, and rules for URL parameters
from pydantic import BaseModel  # base class for describing the shape of the response

from src.predict import forecast, load_model  # forecast function and the cached model loader from Day 5

app = FastAPI(title="Sales Forecasting API")  # create the app; the title shows on the /docs page


class ForecastPoint(BaseModel):  # describes one row of the forecast
    week: str  # the week's date as text, e.g. "2019-01-06"
    forecast: float  # predicted sales for that week


class ForecastResponse(BaseModel):  # describes the full /forecast reply
    sub_category: str  # which sub-category was forecast
    horizon: int  # how many weeks were forecast
    forecast: list[ForecastPoint]  # one ForecastPoint per week


def valid_categories() -> list[str]:  # helper that returns the sub-categories the model knows
    return list(load_model()["categories"])  # read the category list saved inside the joblib bundle


@app.get("/health")  # a GET endpoint at /health
def health():  # runs when someone visits /health
    return {"status": "ok"}  # simple reply to show the API is alive


@app.get("/categories")  # a GET endpoint listing the allowed sub-categories
def categories():  # runs when someone visits /categories
    return {"categories": valid_categories()}  # send back the list so users know what they can ask for


@app.get("/forecast", response_model=ForecastResponse)  # GET /forecast, and its reply must match ForecastResponse
def get_forecast(  # runs when someone calls /forecast
    sub_category: str,  # required text parameter, e.g. Phones
    horizon: int = Query(8, ge=1, le=26),  # weeks ahead: default 8, must be between 1 and 26 or FastAPI returns 422
):
    if sub_category not in valid_categories():  # if the model was never trained on this sub-category
        raise HTTPException(  # stop and send an error reply instead of a forecast
            status_code=400,  # 400 means "bad request" (the caller sent something wrong)
            detail=f"Unknown sub_category '{sub_category}'. Use one of: {valid_categories()}",  # explain what's allowed
        )  # closes HTTPException(
    result = forecast(sub_category, horizon)  # forecast from the saved model (list of dicts or a DataFrame)
    if hasattr(result, "to_dict"):  # if it came back as a pandas DataFrame
        result = result.to_dict(orient="records")  # turn it into a list of row dicts
    points = [  # build a clean list the API can send as JSON
        {"week": str(row["week"])[:10], "forecast": float(row["forecast"])}  # date as "YYYY-MM-DD" text, sales as a plain float
        for row in result  # do this for every forecast week
    ]  # closes the list
    return {"sub_category": sub_category, "horizon": horizon, "forecast": points}  # send the reply; FastAPI checks it against ForecastResponse
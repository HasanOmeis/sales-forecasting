import sys  # reads the command-line arguments for the quick test at the bottom
from functools import lru_cache  # remembers a function's result so the model loads only once

import joblib  # loads the saved model bundle

from src.models import recursive_forecast  # Day 3 week-by-week forecasting function
from src.train import MODEL_PATH  # where train_final() saved the model

MAX_HORIZON = 26  # longest forecast allowed (errors grow every week; we validated 8 weeks)


@lru_cache(maxsize=1)  # first call loads the file; later calls reuse it (important for the API)
def load_model(path=MODEL_PATH):  # reads the bundle saved by train_final()
    return joblib.load(path)  # dictionary with model, features, categories, history, ...


def forecast(sub_category, horizon=8):  # forecast one sub-category for the next "horizon" weeks
    bundle = load_model()  # the saved model and its history
    if sub_category not in bundle["categories"]:  # unknown name, e.g. a typo
        raise ValueError(f"unknown sub_category '{sub_category}', choose from {bundle['categories']}")  # clear message for the user
    if not 1 <= horizon <= MAX_HORIZON:  # too short or too long
        raise ValueError(f"horizon must be between 1 and {MAX_HORIZON}")  # clear message for the user
    preds = recursive_forecast(bundle["model"], bundle["history"], horizon)  # forecast all 9 categories (the global model needs them all)
    series = preds[sub_category]  # keep only the one that was asked for
    return [  # one dictionary per forecast week, easy to turn into JSON later
        {"week": d.strftime("%Y-%m-%d"), "forecast": round(float(v), 2)}  # date as text, sales rounded to cents
        for d, v in series.items()  # loop over (date, value) pairs
    ]


if __name__ == "__main__":  # runs only when the file is run directly
    cat = sys.argv[1] if len(sys.argv) > 1 else "Phones"  # first argument: sub-category, default Phones
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # second argument: horizon, default 8
    for row in forecast(cat, h):  # get the forecast
        print(row)  # print one week per line
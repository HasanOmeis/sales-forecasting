from pathlib import Path  # builds file paths that work on Windows and Linux

import mlflow  # experiment tracking library
import matplotlib.pyplot as plt  # used to close figures after logging them
import joblib  # saves and loads Python objects (our model bundle) to a file

from src.data import load_raw, make_weekly_table  # Day 1 loading functions
from src.features import FEATURES  # the 15 feature names the model uses
from src.models import train_lgbm  # Day 3 training function

ROOT = Path(__file__).resolve().parents[1]  # project root folder (one level above src/)
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"  # store runs in mlflow.db at the project root
ARTIFACT_DIR = (ROOT / "mlruns").as_uri()  # store plots and files in mlruns/ at the project root
EXPERIMENT_NAME = "sales-forecasting"  # one experiment holds every run in this project
METRIC_COLS = ["mae", "rmse", "smape", "mase"]  # the four metrics we report


def setup_mlflow():  # points MLflow at our database and experiment
    mlflow.set_tracking_uri(TRACKING_URI)  # same database no matter which folder the code runs from
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:  # first time only: experiment doesn't exist yet
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_DIR)  # create it with our artifact folder
    mlflow.set_experiment(EXPERIMENT_NAME)  # make it the active experiment


def summarize(results_df, cols=METRIC_COLS):  # averages per-fold/per-category scores into one number each
    return {c: float(results_df[c].mean()) for c in cols if c in results_df.columns}  # mean of each metric column present


def log_run(run_name, params, metrics, figure=None, results_df=None):  # logs one model's run to MLflow
    setup_mlflow()  # make sure MLflow is pointed at the right place
    with mlflow.start_run(run_name=run_name):  # open a run; it closes automatically at the end of the block
        mlflow.log_params(params)  # save settings, e.g. model name, horizon, LightGBM params
        mlflow.log_metrics(metrics)  # save scores, e.g. {"mae": 987, "mase": 1.23}
        if figure is not None:  # only if a plot was passed in
            mlflow.log_figure(figure, "forecast_vs_actual.png")  # save the plot as an image in the run
            plt.close(figure)  # free memory so notebooks don't pile up open figures
        if results_df is not None:  # only if the full results table was passed in
            mlflow.log_text(results_df.to_csv(index=False), "results.csv")  # save the per-fold table as a CSV in the run

MODEL_PATH = ROOT / "models" / "lgbm_global.joblib"  # where the final model is saved

TUNED_PARAMS = {  # best LightGBM settings found by Optuna on Day 5
    "n_estimators": 450,  # number of trees
    "learning_rate": 0.024028273084841772,  # step size
    "num_leaves": 24,  # max leaves per tree
    "min_child_samples": 22,  # minimum rows per leaf
    "subsample": 0.9524568394057271,  # share of rows each tree sees
    "subsample_freq": 1,  # re-sample rows for every tree
    "colsample_bytree": 0.6897595228899822,  # share of features each tree sees
    "reg_lambda": 4.397294381099317,  # L2 penalty
    "random_state": 42,  # fixed seed
    "verbose": -1,  # hide LightGBM messages
}


def train_final(params=TUNED_PARAMS, path=MODEL_PATH):  # trains on all data and saves everything predict.py needs
    raw = load_raw(ROOT / "data" / "superstore.csv")  # load and clean the raw CSV
    weekly = make_weekly_table(raw)  # all 209 weeks x 9 sub-categories
    model = train_lgbm(weekly, params)  # train the global model on every week we have
    bundle = {  # one dictionary saved as one file
        "model": model,  # the trained LightGBM model
        "features": FEATURES,  # feature names, in the order the model expects
        "categories": list(weekly.columns),  # the 9 sub-categories it can forecast
        "params": params,  # the settings it was trained with
        "history": weekly,  # past weekly sales, needed to build lags when forecasting
        "trained_until": str(weekly.index[-1].date()),  # last week of training data
    }
    path.parent.mkdir(exist_ok=True)  # create the models/ folder if it doesn't exist
    joblib.dump(bundle, path)  # write the bundle to models/lgbm_global.joblib
    return bundle  # hand it back so the caller can check it


if __name__ == "__main__":  # runs only when this file is run directly
    b = train_final()  # train and save the final model
    print("saved to", MODEL_PATH)  # where it went
    print("trained until", b["trained_until"], "| categories:", len(b["categories"]))  # quick check
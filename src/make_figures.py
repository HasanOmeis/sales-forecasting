# Load pandas to read the results CSV files
import pandas as pd
# Load matplotlib's plotting module and call it plt
import matplotlib.pyplot as plt
# Reuse your own functions that load the data and build the weekly table
from src.data import load_raw, make_weekly_table


# Define a function that plots total weekly sales across the 9 sub-categories
def plot_weekly_sales(weekly):
    # Add up the 9 columns so each week has one total
    total = weekly.sum(axis=1)
    # Create a wide figure and one set of axes
    fig, ax = plt.subplots(figsize=(11, 4))
    # Draw the weekly totals as a thin line
    ax.plot(total.index, total.values, linewidth=1)
    # Draw a smoother 8-week rolling average on top to show the trend
    ax.plot(total.index, total.rolling(8).mean(), linewidth=2, label="8-week average")
    # Give the chart a title
    ax.set_title("Total weekly sales, 9 sub-categories (2015-2018)")
    # Label the y axis
    ax.set_ylabel("Sales ($)")
    # Show the legend for the rolling average line
    ax.legend()
    # Remove extra white space around the chart
    fig.tight_layout()
    # Save the figure as a PNG in results/
    fig.savefig("results/weekly_sales.png", dpi=150)
    # Close the figure to free memory
    plt.close(fig)


# Define a function that plots average monthly sales to show seasonality
def plot_monthly_seasonality(weekly):
    # Add up the 9 columns so each week has one total
    total = weekly.sum(axis=1)
    # Add up the weeks into calendar months ("ME" means month end)
    monthly = total.resample("ME").sum()
    # Average each calendar month over the 4 years (all Januaries, all Februaries...)
    by_month = monthly.groupby(monthly.index.month).mean()
    # Create a figure and one set of axes
    fig, ax = plt.subplots(figsize=(9, 4))
    # Draw one bar per month
    ax.bar(by_month.index, by_month.values)
    # Show the month numbers 1 to 12 on the x axis
    ax.set_xticks(range(1, 13))
    # Give the chart a title
    ax.set_title("Average monthly sales: peaks in September, November and December")
    # Label the x axis
    ax.set_xlabel("Month")
    # Label the y axis
    ax.set_ylabel("Average sales ($)")
    # Remove extra white space around the chart
    fig.tight_layout()
    # Save the figure as a PNG in results/
    fig.savefig("results/monthly_seasonality.png", dpi=150)
    # Close the figure to free memory
    plt.close(fig)


# Define a function that compares every model's MASE in one bar chart
def plot_model_comparison():
    # Read the average scores of the baselines and the first LightGBM models
    scores = pd.read_csv("results/all_scores.csv")
    # Read the tuned model's scores per fold and category
    tuned = pd.read_csv("results/tuned_scores_by_fold.csv")
    # Build one row with the tuned model's average MASE
    tuned_row = pd.DataFrame({"model": ["lgbm_global_tuned"], "mase": [tuned["mase"].mean()]})
    # Stack the tuned row under the other models
    table = pd.concat([scores[["model", "mase"]], tuned_row], ignore_index=True)
    # Sort so the best model (lowest MASE) ends up at the top of the chart
    table = table.sort_values("mase", ascending=False)
    # Colour LightGBM models in one colour and baselines in grey
    colors = ["tab:blue" if m.startswith("lgbm") else "tab:gray" for m in table["model"]]
    # Create a figure and one set of axes
    fig, ax = plt.subplots(figsize=(9, 4))
    # Draw horizontal bars, one per model
    ax.barh(table["model"], table["mase"], color=colors)
    # Draw a dashed line at MASE 1, the level of a simple naive forecast
    ax.axvline(1.0, linestyle="--", color="black", linewidth=1)
    # Write each MASE value at the end of its bar
    for i, v in enumerate(table["mase"]):
        # Place the number just right of the bar, 2 decimals
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
            # Leave room on the right so the last number isn't cut off
    ax.set_xlim(0, table["mase"].max() + 0.15)
    # Give the chart a title
    ax.set_title("MASE by model, 5 walk-forward folds x 9 sub-categories (lower is better)")
    # Label the x axis
    ax.set_xlabel("MASE")
    # Remove extra white space around the chart
    fig.tight_layout()
    # Save the figure as a PNG in results/
    fig.savefig("results/model_comparison.png", dpi=150)
    # Close the figure to free memory
    plt.close(fig)


# Run the lines below only when this file is run directly
if __name__ == "__main__":
    # Build the weekly table from the raw CSV
    weekly = make_weekly_table(load_raw("data/superstore.csv"))
    # Make the weekly sales figure
    plot_weekly_sales(weekly)
    # Make the monthly seasonality figure
    plot_monthly_seasonality(weekly)
    # Make the model comparison figure
    plot_model_comparison()
    # Confirm that everything was saved
    print("Saved 3 figures to results/")
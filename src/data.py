# Load the pandas library and give it the short name "pd"
import pandas as pd

# The 9 sub-categories we chose to model on Day 1, kept in one place so every file uses the same list
TOP_SUBCATEGORIES = ["Phones", "Chairs", "Storage", "Binders", "Accessories", "Appliances", "Furnishings", "Paper", "Tables"]


# Define a function that loads and cleans the raw Superstore CSV
def load_raw(path):
    # Read the CSV file at that path into a table (DataFrame) called df
    df = pd.read_csv(path)

    # Convert the text dates to real dates (day/month/year format)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

    # List every column except Row ID, which is unique and would hide duplicates
    cols = [c for c in df.columns if c != "Row ID"]

    # Drop rows that repeat in all of those columns, keeping the first copy
    df = df.drop_duplicates(subset=cols)

    # Send the cleaned table back to whoever called the function
    return df


# Define a function that builds the weekly sales table, one column per sub-category
def make_weekly_table(df, categories=TOP_SUBCATEGORIES):
    # Group sales by sub-category and by week, adding up the sales in each group
    weekly = df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="W")])["Sales"].sum()

    # Turn the sub-categories into columns, so each column is one weekly series
    weekly = weekly.unstack(level=0)

    # Build the full list of weeks from the first to the last order week
    all_weeks = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W")

    # Make sure every week appears, and fill weeks with no sales with 0
    weekly = weekly.reindex(all_weeks).fillna(0)

    # If a list of categories was given, keep only those columns, in that order
    if categories is not None:
        # Select the chosen columns; a misspelled name raises an error instead of silently disappearing
        weekly = weekly[categories]

    # Send the weekly table back to whoever called the function
    return weekly


# Run the lines below only when you run this file directly
if __name__ == "__main__":
    # Call the function with the path to your CSV and save the result in "data"
    data = load_raw("data/superstore.csv")
    # Print the first 5 rows to check that it worked
    print(data.head())
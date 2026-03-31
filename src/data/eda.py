# src/data/eda.py
# Exploratory Data Analysis — understand the data before modeling.
# This is NOT for the ML model. It's for US — to make smart decisions
# about features, seasonality, and model choice.

import pandas as pd

def run_eda(df: pd.DataFrame) -> None:
    """
    Prints key insights about the cleaned daily sales DataFrame.

    Args:
        df: cleaned daily sales DataFrame from cleaner.py
    """

    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    # --- Basic shape ---
    print(f"\n Total days in dataset : {len(df)}")
    print(f" Date range            : {df['order_date'].min()} → {df['order_date'].max()}")
    print(f" Total revenue         : ${df['total_sales'].sum():,.2f}")
    print(f" Total profit          : ${df['total_profit'].sum():,.2f}")

    # --- Sales statistics ---
    print("\n--- Daily Sales Statistics ---")
    print(df["total_sales"].describe().round(2))

    # --- Add time columns temporarily for analysis ---
    # We're not saving these — just using them to understand patterns
    temp = df.copy()
    temp["year"] = temp["order_date"].dt.year
    temp["month"] = temp["order_date"].dt.month
    temp["month_name"] = temp["order_date"].dt.strftime("%b")  # Jan, Feb...
    temp["day_of_week"] = temp["order_date"].dt.day_name()

    # --- Sales by year ---
    print("\n--- Total Sales by Year ---")
    yearly = temp.groupby("year")["total_sales"].sum().round(2)
    print(yearly.to_string())

    # --- Sales by month (averaged across all years) ---
    print("\n--- Average Daily Sales by Month ---")
    monthly = (
        temp.groupby(["month", "month_name"])["total_sales"]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("month")
    )
    for _, row in monthly.iterrows():
        bar = "█" * int(row["total_sales"] / 50)  # simple text bar chart
        print(f"  {row['month_name']:>3}: ${row['total_sales']:>8.2f}  {bar}")

   
    print("\n--- Top 5 Best Sales Days ---")
    print(df.nlargest(5, "total_sales")[["order_date", "total_sales"]].to_string(index=False))

    print("\n--- Top 5 Worst Sales Days ---")
    print(df.nsmallest(5, "total_sales")[["order_date", "total_sales"]].to_string(index=False))

    # --- Day of week pattern ---
    print("\n--- Average Sales by Day of Week ---")
    dow = temp.groupby("day_of_week")["total_sales"].mean().round(2)
    print(dow.to_string())

    full_range = pd.date_range(
        start=df["order_date"].min(),
        end=df["order_date"].max(),
        freq="D"
    )
    missing_dates = full_range.difference(df["order_date"])
    print(f"\n--- Missing Dates ---")
    print(f"  Days with no orders: {len(missing_dates)} out of {len(full_range)} total days")

    print("\n" + "="*50)
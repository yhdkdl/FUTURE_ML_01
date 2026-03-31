# src/data/eda.py
# Exploratory Data Analysis on weekly aggregated sales data.

import pandas as pd

def run_eda(df: pd.DataFrame) -> None:
    """
    Prints key insights about the cleaned weekly sales DataFrame.

    Args:
        df: cleaned weekly sales DataFrame from cleaner.py
    """

    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    # --- Basic shape ---
    print(f"\n  Total weeks in dataset : {len(df)}")
    print(f"  Date range             : {df['week_start'].min().date()} → {df['week_start'].max().date()}")
    print(f"  Total revenue          : ${df['total_sales'].sum():,.2f}")
    print(f"  Total profit           : ${df['total_profit'].sum():,.2f}")

    # --- Sales statistics ---
    print("\n--- Weekly Sales Statistics ---")
    print(df["total_sales"].describe().round(2))

    # --- Add time columns for analysis ---
    temp = df.copy()
    temp["year"]       = temp["week_start"].dt.year
    temp["month"]      = temp["week_start"].dt.month
    temp["month_name"] = temp["week_start"].dt.strftime("%b")
    temp["quarter"]    = temp["week_start"].dt.quarter

    # --- Sales by year ---
    print("\n--- Total Sales by Year ---")
    yearly = temp.groupby("year")["total_sales"].sum().round(2)
    print(yearly.to_string())

    # --- Sales by month (averaged across all years) ---
    print("\n--- Average Weekly Sales by Month ---")
    monthly = (
        temp.groupby(["month", "month_name"])["total_sales"]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("month")
    )
    for _, row in monthly.iterrows():
        bar = "█" * int(row["total_sales"] / 200)
        print(f"  {row['month_name']:>3}: ${row['total_sales']:>9.2f}  {bar}")

    # --- Best and worst weeks ---
    print("\n--- Top 5 Best Sales Weeks ---")
    print(
        df.nlargest(5, "total_sales")[["week_start", "total_sales"]]
        .to_string(index=False)
    )

    print("\n--- Top 5 Worst Sales Weeks ---")
    print(
        df.nsmallest(5, "total_sales")[["week_start", "total_sales"]]
        .to_string(index=False)
    )

    # --- Quarter breakdown ---
    print("\n--- Average Weekly Sales by Quarter ---")
    quarterly = temp.groupby("quarter")["total_sales"].mean().round(2)
    for q, val in quarterly.items():
        bar = "█" * int(val / 200)
        print(f"  Q{q}: ${val:>9.2f}  {bar}")

    print("\n" + "="*50)
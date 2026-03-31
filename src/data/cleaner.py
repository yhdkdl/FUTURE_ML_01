

import os
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Superstore DataFrame.

    Steps:
        1. Standardize column names
        2. Parse dates
        3. Drop duplicates
        4. Handle missing values
        5. Ensure correct data types
        6. Aggregate to daily sales (our forecasting target)

    Args:
        df: raw DataFrame from loader

    Returns:
        Cleaned DataFrame ready for feature engineering
    """

    print("\n[cleaner] Starting cleaning pipeline...")

    # --- Step 1: Standardize column names ---
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    print(f"[cleaner] Standardized column names: {list(df.columns)}")

    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=False)
    df["ship_date"] = pd.to_datetime(df["ship_date"], dayfirst=False)
    print(f"[cleaner] Date range: {df['order_date'].min()} to {df['order_date'].max()}")

    # --- Step 3: Drop exact duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[cleaner] Dropped {before - after} duplicate rows")

    # --- Step 4: Handle missing values ---
    missing = df.isnull().sum()
    missing = missing[missing > 0]  # only show columns that have nulls
    if len(missing) > 0:
        print(f"[cleaner] Missing values found:\n{missing}")
        # Drop rows where sales is null — we can't forecast without a target value
        df = df.dropna(subset=["sales"])
        # Fill any remaining nulls in text columns with "Unknown"
        text_cols = df.select_dtypes(include="object").columns
        df[text_cols] = df[text_cols].fillna("Unknown")
    else:
        print("[cleaner] No missing values found")

    # --- Step 5: Ensure correct data types ---
    # Sales and profit must be floats for math operations
    df["sales"] = df["sales"].astype(float)
    df["profit"] = df["profit"].astype(float)
    df["quantity"] = df["quantity"].astype(int)

    # --- Step 6: Aggregate to WEEKLY total sales ---
    # Daily retail data is too noisy for reliable forecasting.
    # Weekly aggregation smooths variance and matches how real
    # businesses actually plan (by week, not by day).
    # W-MON = week ending Monday, giving us clean 7-day buckets.
    df = df.set_index("order_date")

    weekly_sales = (
        df.resample("W-MON")   # group into weekly buckets ending Monday
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            num_orders=("order_id", "nunique"),
            num_items=("quantity", "sum")
        )
        .reset_index()
        .rename(columns={"order_date": "week_start"})
        .sort_values("week_start")
    )

    # Drop any weeks with zero sales entirely
    # These are incomplete boundary weeks at dataset edges
    weekly_sales = weekly_sales[weekly_sales["total_sales"] > 0].reset_index(drop=True)

    print(f"[cleaner] Aggregated to {len(weekly_sales)} weekly records")
    print(f"[cleaner] Columns in clean data: {list(weekly_sales.columns)}")

    return weekly_sales


def save_processed(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the cleaned weekly data to disk.

    Args:
        df: cleaned weekly DataFrame
        filepath: output CSV path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[cleaner] Saved processed data to: {filepath}")
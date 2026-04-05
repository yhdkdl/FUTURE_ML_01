

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


    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[cleaner] Dropped {before - after} duplicate rows")

    missing = df.isnull().sum()
    missing = missing[missing > 0] 
    if len(missing) > 0:
        print(f"[cleaner] Missing values found:\n{missing}")
      
        df = df.dropna(subset=["sales"])
     
        text_cols = df.select_dtypes(include="object").columns
        df[text_cols] = df[text_cols].fillna("Unknown")
    else:
        print("[cleaner] No missing values found")


    df["sales"] = df["sales"].astype(float)
    df["profit"] = df["profit"].astype(float)
    df["quantity"] = df["quantity"].astype(int)

  
    df = df.set_index("order_date")

    weekly_sales = (
        df.resample("W-MON")  
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



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

    # --- Step 6: Aggregate to daily total sales ---
    daily_sales = (
        df.groupby("order_date")
        .agg(
            total_sales=("sales", "sum"),       # sum all sales for that day
            total_profit=("profit", "sum"),     # sum all profit for that day
            num_orders=("order_id", "nunique"), # count unique orders
            num_items=("quantity", "sum")       # total items sold
        )
        .reset_index()  # turn order_date back into a column (not an index)
        .sort_values("order_date")  # ensure chronological order
    )

    print(f"[cleaner] Aggregated to {len(daily_sales)} daily records")
    print(f"[cleaner] Columns in clean data: {list(daily_sales.columns)}")

    return daily_sales


def save_processed(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the cleaned DataFrame to the processed data folder.

    Args:
        df: cleaned DataFrame
        filepath: destination path
    """
    df.to_csv(filepath, index=False)
    print(f"\n[cleaner] Saved processed data to: {filepath}")
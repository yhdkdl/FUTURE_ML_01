# src/features/engineer.py
# Transforms clean daily sales data into ML-ready features.
# Rule: this file only ADDS columns — it never removes or modifies existing ones.

import pandas as pd
import numpy as np
import os

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the cleaned daily sales DataFrame and engineers
    all time-based features needed for forecasting.

    Args:
        df: cleaned daily sales DataFrame (output of cleaner.py)

    Returns:
        Feature-rich DataFrame ready for model training
    """

    print("\n[engineer] Starting feature engineering...")

    # Work on a copy — never mutate the input DataFrame
    # This keeps each pipeline stage independent and debuggable
    data = df.copy()

    # Ensure order_date is datetime (it should be, but be defensive)
    data["order_date"] = pd.to_datetime(data["order_date"])

    # Sort chronologically — critical for lag/rolling features
    # If dates are out of order, lag features will be wrong
    data = data.sort_values("order_date").reset_index(drop=True)

    # -------------------------------------------------------
    # BLOCK 1: Calendar Features
    # These extract structured information from the date itself
    # -------------------------------------------------------

    # Year — captures the growth trend over time
    # e.g. 2014=0, 2015=1 so the model sees time moving forward
    data["year"] = data["order_date"].dt.year

    # Month (1-12) — captures seasonality within a year
    data["month"] = data["order_date"].dt.month

    # Quarter (1-4) — captures business quarter patterns
    data["quarter"] = data["order_date"].dt.quarter

    # Day of week (0=Monday, 6=Sunday) — weekly rhythm
    data["day_of_week"] = data["order_date"].dt.dayofweek

    # Day of month (1-31) — pay cycle / end-of-month effects
    data["day_of_month"] = data["order_date"].dt.day

    # Week of year (1-52) — retail planning cycles
    data["week_of_year"] = data["order_date"].dt.isocalendar().week.astype(int)

    print("[engineer] ✓ Calendar features added")

    # -------------------------------------------------------
    # BLOCK 2: Boolean Flag Features
    # Binary signals (0 or 1) for specific business events
    # -------------------------------------------------------

    # Is weekend? (Saturday=5, Sunday=6)
    # Retail behaves very differently on weekends
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Is last day of month?
    # End-of-month often sees purchase surges (budgets, deadlines)
    data["is_month_end"] = data["order_date"].dt.is_month_end.astype(int)

    # Is last day of quarter?
    # B2B companies rush purchases before quarter closes
    data["is_quarter_end"] = data["order_date"].dt.is_quarter_end.astype(int)

    print("[engineer] ✓ Boolean flag features added")

    # -------------------------------------------------------
    # BLOCK 3: Reindex to fill missing dates
    # The Superstore dataset has days with zero orders (weekends, holidays)
    # Lag and rolling features REQUIRE a continuous daily sequence
    # Without this, "7 days ago" might actually be 10 calendar days ago
    # -------------------------------------------------------

    full_date_range = pd.date_range(
        start=data["order_date"].min(),
        end=data["order_date"].max(),
        freq="D"  # D = calendar day frequency
    )

    # Set order_date as index so we can reindex by date
    data = data.set_index("order_date")

    # Reindex — inserts rows for missing dates with NaN values
    data = data.reindex(full_date_range)

    # Fill missing sales with 0 (no orders = zero sales that day)
    # Fill other numeric columns with 0 too
    data["total_sales"] = data["total_sales"].fillna(0)
    data["total_profit"] = data["total_profit"].fillna(0)
    data["num_orders"] = data["num_orders"].fillna(0)
    data["num_items"] = data["num_items"].fillna(0)

    # Rename the index back to order_date and reset it to a column
    data.index.name = "order_date"
    data = data.reset_index()

    # Re-extract calendar features for the newly inserted dates
    # (the gap-fill rows have NaN for these from the reindex)
    data["year"] = data["order_date"].dt.year
    data["month"] = data["order_date"].dt.month
    data["quarter"] = data["order_date"].dt.quarter
    data["day_of_week"] = data["order_date"].dt.dayofweek
    data["day_of_month"] = data["order_date"].dt.day
    data["week_of_year"] = data["order_date"].dt.isocalendar().week.astype(int)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_month_end"] = data["order_date"].dt.is_month_end.astype(int)
    data["is_quarter_end"] = data["order_date"].dt.is_quarter_end.astype(int)

    print(f"[engineer] ✓ Date gaps filled — {len(data)} continuous daily rows")

    # -------------------------------------------------------
    # BLOCK 4: Lag Features
    # "What were sales N days ago?"
    # These are the single most powerful features for forecasting
    # The model learns: "last week's sales predict this week's sales"
    # -------------------------------------------------------

    # shift(N) moves the sales column DOWN by N rows
    # So row[today] gets the value that was at row[today - N]
    data["lag_7"] = data["total_sales"].shift(7)   # same day last week
    data["lag_14"] = data["total_sales"].shift(14)  # two weeks ago
    data["lag_30"] = data["total_sales"].shift(30)  # approx one month ago

    print("[engineer] ✓ Lag features added (7, 14, 30 days)")

    # -------------------------------------------------------
    # BLOCK 5: Rolling Window Features
    # "What was the average/std sales over the last N days?"
    # Smooths out daily noise and captures the recent trend
    # -------------------------------------------------------

    # rolling(N) creates a window of the last N rows
    # .mean() averages them, .std() measures their spread
    # min_periods=1 means: start calculating even if window isn't full yet
    data["rolling_mean_7"] = (
        data["total_sales"]
        .shift(1)                    # shift(1) so today's value isn't included
        .rolling(window=7, min_periods=1)
        .mean()
        .round(2)
    )

    data["rolling_mean_30"] = (
        data["total_sales"]
        .shift(1)
        .rolling(window=30, min_periods=1)
        .mean()
        .round(2)
    )

    data["rolling_std_7"] = (
        data["total_sales"]
        .shift(1)
        .rolling(window=7, min_periods=1)
        .std()
        .round(2)
    )

    print("[engineer] ✓ Rolling features added (mean_7, mean_30, std_7)")

    # -------------------------------------------------------
    # BLOCK 6: Drop rows where lag features are NaN
    # The first 30 rows can't have lag_30 values (nothing before them)
    # Keeping NaN rows would corrupt the model training
    # -------------------------------------------------------

    before = len(data)
    data = data.dropna(subset=["lag_7", "lag_14", "lag_30"])
    after = len(data)
    print(f"[engineer] ✓ Dropped {before - after} rows with NaN lag values")

    # Final shape report
    print(f"[engineer] Final dataset: {len(data)} rows × {len(data.columns)} columns")
    print(f"[engineer] Features: {list(data.columns)}")

    return data


def save_features(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the feature-engineered DataFrame to disk.

    Args:
        df: feature-rich DataFrame
        filepath: destination path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[engineer] Saved feature data to: {filepath}")
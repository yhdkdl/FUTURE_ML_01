# src/features/engineer.py
# Transforms clean weekly sales data into ML-ready features.
# Weekly granularity gives us smoother, more learnable patterns.

import pandas as pd
import numpy as np
import os

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers time-based features from weekly sales data.

    Args:
        df: cleaned weekly sales DataFrame (output of cleaner.py)

    Returns:
        Feature-rich DataFrame ready for model training
    """

    print("\n[engineer] Starting feature engineering...")

    data = df.copy()

    # Ensure week_start is datetime
    data["week_start"] = pd.to_datetime(data["week_start"])

    # Sort chronologically — critical for lag/rolling features
    data = data.sort_values("week_start").reset_index(drop=True)

    # -------------------------------------------------------
    # BLOCK 1: Calendar Features
    # -------------------------------------------------------

    data["year"]         = data["week_start"].dt.year
    data["month"]        = data["week_start"].dt.month
    data["quarter"]      = data["week_start"].dt.quarter
    data["week_of_year"] = data["week_start"].dt.isocalendar().week.astype(int)
    data["day_of_month"] = data["week_start"].dt.day

    # Encode year as a relative integer so the model sees time progression
    # e.g. 2014→0, 2015→1, 2016→2, 2017→3
    data["year_index"] = data["year"] - data["year"].min()

    print("[engineer] ✓ Calendar features added")

    # -------------------------------------------------------
    # BLOCK 2: Boolean Flag Features
    # -------------------------------------------------------

    # Is this week in Q4? (Oct, Nov, Dec — holiday season)
    # Q4 is the highest sales quarter for most retailers
    data["is_q4"] = (data["quarter"] == 4).astype(int)

    # Is this week at the end of a quarter?
    # B2B purchasing surges at quarter-end
    data["is_quarter_end"] = data["week_start"].dt.is_quarter_end.astype(int)

    # Is this week in the back-to-school period? (Aug-Sep)
    data["is_bts_season"] = (data["month"].isin([8, 9])).astype(int)

    print("[engineer] ✓ Boolean flag features added")

    # -------------------------------------------------------
    # BLOCK 3: Lag Features
    # For weekly data, we look back 1, 2, and 4 weeks
    # shift(1) = last week, shift(4) = approx 1 month ago
    # -------------------------------------------------------

    data["lag_1"]  = data["total_sales"].shift(1)   # last week
    data["lag_2"]  = data["total_sales"].shift(2)   # 2 weeks ago
    data["lag_4"]  = data["total_sales"].shift(4)   # ~1 month ago
    data["lag_52"] = data["total_sales"].shift(52)  # same week last year

    print("[engineer] ✓ Lag features added (1, 2, 4, 52 weeks)")

    # -------------------------------------------------------
    # BLOCK 4: Rolling Window Features
    # shift(1) before rolling so current week isn't included
    # -------------------------------------------------------

    data["rolling_mean_4"] = (
        data["total_sales"]
        .shift(1)
        .rolling(window=4, min_periods=1)
        .mean()
        .round(2)
    )

    data["rolling_mean_12"] = (
        data["total_sales"]
        .shift(1)
        .rolling(window=12, min_periods=1)
        .mean()
        .round(2)
    )

    data["rolling_std_4"] = (
        data["total_sales"]
        .shift(1)
        .rolling(window=4, min_periods=1)
        .std()
        .fillna(0)  # std is NaN when window has only 1 value
        .round(2)
    )

    print("[engineer] ✓ Rolling features added (mean_4, mean_12, std_4)")

    # -------------------------------------------------------
    # BLOCK 5: Drop rows where lag features are NaN
    # lag_52 requires 52 weeks of history — first year will be dropped
    # This is expected and acceptable
    # -------------------------------------------------------

    before = len(data)

    # We drop on lag_4 only — lag_52 would remove too much data
    # We'll fill lag_52 NaNs with the rolling mean instead
    data["lag_52"] = data["lag_52"].fillna(data["rolling_mean_12"])

    data = data.dropna(subset=["lag_1", "lag_2", "lag_4"])
    after = len(data)

    print(f"[engineer] ✓ Dropped {before - after} rows with NaN lag values")
    print(f"[engineer] Final dataset: {len(data)} rows × {len(data.columns)} columns")
    print(f"[engineer] Features: {list(data.columns)}")

    return data


def save_features(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the feature-engineered DataFrame to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[engineer] Saved feature data to: {filepath}")
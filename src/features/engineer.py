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

    data["week_start"] = pd.to_datetime(data["week_start"])

    data = data.sort_values("week_start").reset_index(drop=True)

  

    data["year"]         = data["week_start"].dt.year
    data["month"]        = data["week_start"].dt.month
    data["quarter"]      = data["week_start"].dt.quarter
    data["week_of_year"] = data["week_start"].dt.isocalendar().week.astype(int)
    data["day_of_month"] = data["week_start"].dt.day

    data["year_index"] = data["year"] - data["year"].min()

    print("[engineer] ✓ Calendar features added")

    data["is_q4"] = (data["quarter"] == 4).astype(int)

    data["is_quarter_end"] = data["week_start"].dt.is_quarter_end.astype(int)

    data["is_bts_season"] = (data["month"].isin([8, 9])).astype(int)

    print("[engineer] ✓ Boolean flag features added")

    data["lag_1"]  = data["total_sales"].shift(1)  
    data["lag_2"]  = data["total_sales"].shift(2)   
    data["lag_4"]  = data["total_sales"].shift(4)  
    data["lag_52"] = data["total_sales"].shift(52) 

    print("[engineer] ✓ Lag features added (1, 2, 4, 52 weeks)")

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
        .fillna(0)  
        .round(2)
    )

    print("[engineer] ✓ Rolling features added (mean_4, mean_12, std_4)")


    before = len(data)

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

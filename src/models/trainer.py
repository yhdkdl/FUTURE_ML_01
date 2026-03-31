# src/models/trainer.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLUMNS = [
    "year_index",
    "month",
    "quarter",
    "week_of_year",
    "day_of_month",
    "is_q4",
    "is_quarter_end",
    "is_bts_season",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_12",
    "rolling_std_4",
]

TARGET_COLUMN = "total_sales"


def split_data(df: pd.DataFrame, test_weeks_per_year: int = 10):
    

    df = df.copy()

    # For each year, find the last N week numbers
    # These become our test set — one block per year
    test_mask = df.groupby("year")["week_of_year"].transform(
        lambda weeks: weeks >= weeks.quantile(1 - test_weeks_per_year / 52)
    )

    train = df[~test_mask].reset_index(drop=True)  # everything NOT in test
    test  = df[test_mask].reset_index(drop=True)   # the held-out weeks

    # split_date = earliest date in the test set (for visualization reference)
    split_date = test["week_start"].min()

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test  = test[FEATURE_COLUMNS]
    y_test  = test[TARGET_COLUMN]

    print(f"\n[trainer] Train size : {len(X_train)} weeks")
    print(f"[trainer] Test size  : {len(X_test)} weeks")
    print(f"[trainer] Test years covered : {sorted(test['year'].unique())}")
    print(f"[trainer] Split ref date     : {split_date.date()}")

    return X_train, X_test, y_train, y_test, split_date

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Trains a Random Forest Regressor on weekly training data.
    """

    print("\n[trainer] Training Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=300,    # more trees for weekly data stability
        max_depth=8,         # slightly shallower — weekly data is less noisy
        min_samples_leaf=3,  # weekly dataset is smaller so lower minimum
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("[trainer] ✓ Model training complete")

    return model


def save_model(model: RandomForestRegressor, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[trainer] ✓ Model saved to: {filepath}")


def load_model(filepath: str) -> RandomForestRegressor:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at: {filepath}")
    model = joblib.load(filepath)
    print(f"[trainer] ✓ Model loaded from: {filepath}")
    return model
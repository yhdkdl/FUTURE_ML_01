
import pandas as pd
import numpy as np
from src.models.trainer import FEATURE_COLUMNS


def generate_future_weeks(last_date: pd.Timestamp, weeks: int = 12) -> pd.DataFrame:
    """
    Creates a DataFrame of future weekly dates.

    Args:
        last_date: last week_start date in historical data
        weeks: how many weeks ahead to forecast (default 12 = 3 months)

    Returns:
        DataFrame with future week_start dates and calendar features
    """

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=weeks,
        freq="W-MON"  
    )

    future_df = pd.DataFrame({"week_start": future_dates})

    future_df["year"]         = future_df["week_start"].dt.year
    future_df["month"]        = future_df["week_start"].dt.month
    future_df["quarter"]      = future_df["week_start"].dt.quarter
    future_df["week_of_year"] = future_df["week_start"].dt.isocalendar().week.astype(int)
    future_df["day_of_month"] = future_df["week_start"].dt.day


    future_df["year_index"]   = future_df["year"] - 2014

    future_df["is_q4"]          = (future_df["quarter"] == 4).astype(int)
    future_df["is_quarter_end"] = future_df["week_start"].dt.is_quarter_end.astype(int)
    future_df["is_bts_season"]  = (future_df["month"].isin([8, 9])).astype(int)

    return future_df


def generate_forecast(
    model,
    historical_df: pd.DataFrame,
    weeks: int = 12
) -> pd.DataFrame:
    """
    Recursive multi-step weekly forecast.
    Each predicted week feeds into the next week's lag features.

    Args:
        model: trained RandomForestRegressor
        historical_df: full feature-engineered weekly DataFrame
        weeks: number of weeks to forecast

    Returns:
        DataFrame with future week_start dates and predicted_sales
    """

    print(f"\n[predictor] Generating {weeks}-week forecast...")

   
    known_sales = list(historical_df["total_sales"].values)

    last_date = historical_df["week_start"].max()
    future_df = generate_future_weeks(last_date, weeks)

    predictions = []

    for i in range(weeks):

        lag_1  = known_sales[-1]  if len(known_sales) >= 1  else 0
        lag_2  = known_sales[-2]  if len(known_sales) >= 2  else 0
        lag_4  = known_sales[-4]  if len(known_sales) >= 4  else 0
        lag_52 = known_sales[-52] if len(known_sales) >= 52 else np.mean(known_sales)

        # Rolling features
        rolling_mean_4  = np.mean(known_sales[-4:])  if len(known_sales) >= 4  else np.mean(known_sales)
        rolling_mean_12 = np.mean(known_sales[-12:]) if len(known_sales) >= 12 else np.mean(known_sales)
        rolling_std_4   = np.std(known_sales[-4:])   if len(known_sales) >= 4  else 0

        row = future_df.iloc[i]

        feature_row = pd.DataFrame([{
            "year_index":      row["year_index"],
            "month":           row["month"],
            "quarter":         row["quarter"],
            "week_of_year":    row["week_of_year"],
            "day_of_month":    row["day_of_month"],
            "is_q4":           row["is_q4"],
            "is_quarter_end":  row["is_quarter_end"],
            "is_bts_season":   row["is_bts_season"],
            "lag_1":           lag_1,
            "lag_2":           lag_2,
            "lag_4":           lag_4,
            "lag_52":          lag_52,
            "rolling_mean_4":  round(rolling_mean_4, 2),
            "rolling_mean_12": round(rolling_mean_12, 2),
            "rolling_std_4":   round(rolling_std_4, 2),
        }])

        pred = model.predict(feature_row)[0]
        pred = max(0, pred)  

        predictions.append(pred)
        known_sales.append(pred) 

    future_df["predicted_sales"] = np.round(predictions, 2)

    print(f"[predictor] ✓ Forecast complete")
    print(f"[predictor] Forecast period : {future_df['week_start'].min().date()} → {future_df['week_start'].max().date()}")
    print(f"[predictor] Avg predicted weekly sales : ${future_df['predicted_sales'].mean():,.2f}")
    print(f"[predictor] Total predicted revenue    : ${future_df['predicted_sales'].sum():,.2f}")

    return future_df[["week_start", "predicted_sales"]]

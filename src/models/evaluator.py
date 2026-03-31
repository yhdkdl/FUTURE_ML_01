
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Runs the model on test data and calculates evaluation metrics.

    Args:
        model: trained model
        X_test: test features
        y_test: actual sales values for test period

    Returns:
        Dictionary of metric names and values
    """

    print("\n[evaluator] Running model evaluation...")

    # Generate predictions on the test set
    # These are the model's "guesses" for the test period
    predictions = model.predict(X_test)

    # --- MAE: Mean Absolute Error ---
    # Average absolute difference between predicted and actual
    # Easiest metric to explain to a non-technical person
    # "On average, our forecast is off by $X per day"
    mae = mean_absolute_error(y_test, predictions)

    # --- RMSE: Root Mean Squared Error ---
    # Like MAE but squares errors first, then square roots the average
    # Penalizes large errors more heavily than MAE
    # Useful when big errors are especially costly to the business
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

  # --- MAPE: Mean Absolute Percentage Error ---
    # We skip weeks where actual sales = 0 to avoid division by zero.
    # This gives a true percentage error on weeks with real activity.
    actual   = y_test.values
    nonzero  = actual != 0          # boolean mask of valid weeks
    mape = np.mean(
        np.abs((actual[nonzero] - predictions[nonzero]) / actual[nonzero])
    ) * 100
    # --- R² Score ---
    # Measures how much of the sales variance our model explains
    # 1.0 = perfect, 0.0 = no better than predicting the mean every day
    # Negative = worse than predicting the mean (very bad)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE":  round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2":   round(r2, 4)
    }

    return metrics, predictions


def print_evaluation_report(metrics: dict, y_test: pd.Series) -> None:
    """
    Prints a business-friendly evaluation report.

    Args:
        metrics: dictionary of metric values from evaluate_model
        y_test: actual test sales (used to calculate context)
    """

    avg_daily_sales = y_test.mean()

    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)

    print(f"\n  Average actual daily sales : ${avg_daily_sales:,.2f}")
    print(f"  MAE                        : ${metrics['MAE']:,.2f}")
    print(f"  RMSE                       : ${metrics['RMSE']:,.2f}")
    print(f"  MAPE                       : {metrics['MAPE']:.2f}%")
    print(f"  R² Score                   : {metrics['R2']}")

    # Plain English interpretation
    print("\n--- What This Means ---")
    print(f"  On average, our daily forecast is off by ${metrics['MAE']:,.2f}")
    print(f"  That's a {metrics['MAPE']:.1f}% error rate on daily sales")

    # R² interpretation
    if metrics["R2"] >= 0.85:
        verdict = "Excellent — model explains the data very well"
    elif metrics["R2"] >= 0.70:
        verdict = "Good — model captures most patterns"
    elif metrics["R2"] >= 0.50:
        verdict = "Moderate — usable but room for improvement"
    else:
        verdict = "Weak — consider adding more features"

    print(f"  R² verdict: {verdict}")
    print("="*50)


def get_feature_importance(model, feature_columns: list) -> pd.DataFrame:
    """
    Extracts and ranks feature importances from the trained Random Forest.
    Tells us WHICH features the model relied on most.

    Args:
        model: trained RandomForestRegressor
        feature_columns: list of feature names

    Returns:
        DataFrame of features ranked by importance
    """

    importance_df = pd.DataFrame({
        "feature":    feature_columns,
        "importance": model.feature_importances_
    })

    # Sort descending so most important features are at the top
    importance_df = (
        importance_df
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- Feature Importance (Top 10) ---")
    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"  {row['feature']:<20} {row['importance']:.4f}  {bar}")

    return importance_df
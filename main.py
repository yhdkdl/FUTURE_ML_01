# main.py
# Central pipeline — one clean pass, no duplicate stages.

import os
import pandas as pd
from src.data.loader import load_raw_data
from src.data.cleaner import clean_data, save_processed
from src.data.eda import run_eda
from src.features.engineer import build_features, save_features
from src.models.trainer import split_data, train_model, save_model, FEATURE_COLUMNS
from src.models.evaluator import evaluate_model, print_evaluation_report, get_feature_importance
from src.models.predictor import generate_forecast
from src.visualization.charts import (        
    chart_sales_history,                      
    chart_forecast,                           
    chart_seasonality,                        
    chart_model_performance                   
) 

# --- Paths ---
RAW_DATA_PATH       = os.path.join("data", "raw", "superstore.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "weekly_sales.csv")
FEATURES_DATA_PATH  = os.path.join("data", "processed", "features.csv")
MODEL_PATH          = os.path.join("outputs", "models", "forecast_model.joblib")
CHARTS_DIR          = os.path.join("outputs", "charts")
FORECAST_PATH       = os.path.join("data", "processed", "forecast.csv")


def main():
    print("FUTURE_ML_01 — Sales Forecasting System")
    print("="*50)

    # Stage 1: Load
    raw_df = load_raw_data(RAW_DATA_PATH)

    # Stage 2: Clean + aggregate to weekly
    clean_df = clean_data(raw_df)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    save_processed(clean_df, PROCESSED_DATA_PATH)

    # Stage 3: EDA
    run_eda(clean_df)

    # Stage 4: Feature engineering
    features_df = build_features(clean_df)
    save_features(features_df, FEATURES_DATA_PATH)

    # Stage 5: Train/test split
    X_train, X_test, y_train, y_test, split_date = split_data(features_df)

    # Stage 6: Train
    model = train_model(X_train, y_train)
    save_model(model, MODEL_PATH)

    # Stage 7: Evaluate
    metrics, predictions = evaluate_model(model, X_test, y_test)
    print_evaluation_report(metrics, y_test)
    get_feature_importance(model, FEATURE_COLUMNS)

    # Stage 8: Forecast 12 weeks ahead
    forecast_df = generate_forecast(model, features_df, weeks=12)
    forecast_df.to_csv(FORECAST_PATH, index=False)
    print(f"\n[main] Forecast saved to: {FORECAST_PATH}")
    print("\n[main] ✓ Pipeline complete")
 # Stage 9: Visualizations                                         
    print("\n[main] Generating charts...")                            
    chart_sales_history(features_df, CHARTS_DIR)                     
    chart_forecast(features_df, forecast_df, CHARTS_DIR)             
    chart_seasonality(features_df, CHARTS_DIR)                       
    chart_model_performance(                                          
        features_df, model, split_date, metrics, CHARTS_DIR          
    )                                                                 
    print(f"\n[main] ✓ All charts saved to: {CHARTS_DIR}")           
    print("\n[main] ✓ Pipeline complete")

if __name__ == "__main__":
    main()
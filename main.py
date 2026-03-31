
import os
from src.data.loader import load_raw_data
from src.data.cleaner import clean_data, save_processed
from src.data.eda import run_eda

# --- File paths ---
# Using os.path.join keeps this cross-platform (works on Windows, Mac, Linux)
RAW_DATA_PATH = os.path.join("data", "raw", "superstore.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "daily_sales.csv")

def main():
    print("FUTURE_ML_01 — Sales Forecasting System")
    print("="*50)

    # --- Stage 1: Load ---
    raw_df = load_raw_data(RAW_DATA_PATH)

    # --- Stage 2: Clean ---
    clean_df = clean_data(raw_df)

    # --- Stage 3: Save processed data ---
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    save_processed(clean_df, PROCESSED_DATA_PATH)

    # --- Stage 4: EDA ---
    run_eda(clean_df)

if __name__ == "__main__":
    main()
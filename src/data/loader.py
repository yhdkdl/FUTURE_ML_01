# src/data/loader.py
# Responsible for ONE thing only: reading the raw CSV into a DataFrame.
# We never modify data here — that's the cleaner's job.

import pandas as pd
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw Superstore CSV file into a pandas DataFrame.

    Args:
        filepath: path to the raw CSV file

    Returns:
        Raw DataFrame, completely unmodified
    """

    # Check the file actually exists before trying to read it
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    # The Superstore CSV uses latin-1 encoding (not standard UTF-8)
    # Without this, pandas will throw a UnicodeDecodeError
    df = pd.read_csv(filepath, encoding="latin-1")

    print(f"[loader] Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"[loader] Columns: {list(df.columns)}")

    return df
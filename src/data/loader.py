

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

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath, encoding="latin-1")

    print(f"[loader] Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"[loader] Columns: {list(df.columns)}")

    return df

import pandas as pd
import numpy as np
from pathlib import Path
from .config import settings
import logging

log = logging.getLogger("training.data")

def load_training_data(symbol: str, target_col: str = "close", lookforward: int = 1) -> pd.DataFrame:
    """
    Load data from Parquet features, create a target variable.
    """
    symbol_path = settings.features_parquet_dir / symbol
    if not symbol_path.exists():
        raise FileNotFoundError(f"No features data found for {symbol}")
    
    # Read all parquet partitions
    # duckdb is faster for this than reading many files manually with pandas
    # but we need to query the parquet files.
    import duckdb
    df = duckdb.query(f"SELECT * FROM '{symbol_path}/**/*.parquet' ORDER BY ts ASC").to_df()
    
    if df.empty:
        raise ValueError(f"No data rows for {symbol}")

    # Create target: Percent change 'lookforward' steps ahead
    # Simple Example: Predict if price goes up (1) or down (0)
    # Or Predict the next close price. 
    # Let's do simple regression: target is next 'close' price.
    
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in features")

    # Shift target backward to align "current features" with "future price"
    df["target"] = df[target_col].shift(-lookforward)
    
    # Drop rows without target (last N rows)
    df = df.dropna(subset=["target"])
    
    # Drop non-numeric for training (except ts)
    # We keep 'ts' for splitting but drop it for model input usually
    return df

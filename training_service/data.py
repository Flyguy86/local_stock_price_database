import pandas as pd
import numpy as np
from pathlib import Path
from .config import settings
import logging
import duckdb

log = logging.getLogger("training.data")

def get_data_options(symbol: str) -> list[str]:
    symbol_path = settings.features_parquet_dir / symbol
    if not symbol_path.exists():
        return []
    
    try:
        # Check if options column exists first or just try query
        # We select distinct options
        res = duckdb.query(f"SELECT DISTINCT options FROM '{symbol_path}/**/*.parquet'").fetchall()
        # res is list of tuples [(opt_str,), ...]
        return [r[0] for r in res if r[0] is not None]
    except Exception as e:
        log.warning(f"Could not read options for {symbol}: {e}")
        return []

def load_training_data(symbol: str, target_col: str = "close", lookforward: int = 1, options_filter: str = None) -> pd.DataFrame:
    """
    Load data from Parquet features, create a target variable.
    """
    symbol_path = settings.features_parquet_dir / symbol
    if not symbol_path.exists():
        raise FileNotFoundError(f"No features data found for {symbol}")
    
    # Read all parquet partitions
    # duckdb is faster for this than reading many files manually with pandas
    query = f"SELECT * FROM '{symbol_path}/**/*.parquet'"
    
    if options_filter:
        # Escape single quotes in options_filter if any (though json dumps uses double quotes usually)
        safe_filter = options_filter.replace("'", "''")
        query += f" WHERE options = '{safe_filter}'"
    
    query += " ORDER BY ts ASC"
    
    df = duckdb.query(query).to_df()
    
    if df.empty:
        raise ValueError(f"No data rows for {symbol} (filter={options_filter})")

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

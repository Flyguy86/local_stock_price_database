"""
Data loading utilities for Ray Orchestrator.

Uses Ray Data for streaming large datasets without memory overflow.
Integrates with DuckDB/Parquet feature storage.
"""

import logging
from pathlib import Path
from typing import Optional
import tempfile
import shutil

import pandas as pd
import numpy as np
import duckdb
import ray
from ray.data import Dataset

from .config import settings

log = logging.getLogger("ray_orchestrator.data")


def get_available_symbols() -> list[str]:
    """
    Get list of available symbols from feature parquet directory.
    
    Returns:
        List of ticker symbols (e.g., ["AAPL", "GOOGL", "MSFT"])
    """
    parquet_dir = settings.data.features_parquet_dir
    
    if not parquet_dir.exists():
        log.warning(f"Features parquet directory not found: {parquet_dir}")
        return []
    
    symbols = []
    for path in parquet_dir.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            symbols.append(path.name)
    
    return sorted(symbols)


def get_symbol_date_range(symbol: str) -> tuple[str, str]:
    """
    Get the date range available for a symbol.
    
    Args:
        symbol: Ticker symbol
        
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    parquet_dir = settings.data.features_parquet_dir / symbol
    
    if not parquet_dir.exists():
        return ("", "")
    
    dates = []
    for path in parquet_dir.iterdir():
        if path.is_dir() and path.name.startswith("dt="):
            date = path.name.replace("dt=", "")
            dates.append(date)
    
    if not dates:
        return ("", "")
    
    dates.sort()
    return (dates[0], dates[-1])


def load_symbol_data_pandas(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = "1m"
) -> pd.DataFrame:
    """
    Load feature data for a symbol using DuckDB.
    
    This is the traditional pandas-based loader for single trials.
    For distributed loading, use load_symbol_data_ray().
    
    Args:
        symbol: Ticker symbol
        start_date: Start date (YYYY-MM-DD) or None for all
        end_date: End date (YYYY-MM-DD) or None for all  
        timeframe: Resample timeframe (1m, 5m, 15m, 1h, 1d)
        
    Returns:
        DataFrame with features
    """
    parquet_path = settings.data.features_parquet_dir / symbol
    
    if not parquet_path.exists():
        log.warning(f"No data found for symbol: {symbol}")
        return pd.DataFrame()
    
    # Build query with optional date filter
    date_filter = ""
    if start_date:
        date_filter += f" AND dt >= '{start_date}'"
    if end_date:
        date_filter += f" AND dt <= '{end_date}'"
    
    query = f"""
        SELECT * 
        FROM read_parquet('{parquet_path}/*/*.parquet', union_by_name=true)
        WHERE 1=1 {date_filter}
        ORDER BY ts
    """
    
    try:
        df = duckdb.query(query).df()
        log.info(f"Loaded {len(df)} rows for {symbol} ({timeframe})")
        
        # Resample if not 1m
        if timeframe != "1m" and not df.empty:
            df = _resample_dataframe(df, timeframe)
            log.info(f"Resampled to {timeframe}: {len(df)} rows")
        
        return df
        
    except Exception as e:
        log.error(f"Failed to load data for {symbol}: {e}")
        return pd.DataFrame()


def _resample_dataframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample DataFrame to target timeframe."""
    if "ts" not in df.columns:
        return df
    
    df = df.set_index("ts").sort_index()
    
    # Build aggregation dictionary
    agg_dict = {}
    for col in df.columns:
        if col == "open":
            agg_dict[col] = "first"
        elif col == "high":
            agg_dict[col] = "max"
        elif col == "low":
            agg_dict[col] = "min"
        elif col == "close":
            agg_dict[col] = "last"
        elif col == "volume":
            agg_dict[col] = "sum"
        elif col == "trade_count":
            agg_dict[col] = "sum"
        elif col == "data_split":
            # Conservative: if any test in bucket, mark as test
            agg_dict[col] = lambda x: "test" if "test" in set(x.astype(str)) else "train"
        else:
            agg_dict[col] = "last"
    
    df = df.resample(timeframe).agg(agg_dict)
    df = df.dropna(subset=["close"])
    df = df.reset_index()
    
    return df


@ray.remote
def load_symbol_data_remote(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = "1m"
) -> pd.DataFrame:
    """
    Ray remote version of data loader.
    
    This allows parallel loading of multiple symbols across workers.
    """
    return load_symbol_data_pandas(symbol, start_date, end_date, timeframe)


def load_multi_symbol_data_ray(
    symbols: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: str = "1m"
) -> dict[str, pd.DataFrame]:
    """
    Load data for multiple symbols in parallel using Ray.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Resample timeframe
        
    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    # Launch parallel loads
    futures = {
        symbol: load_symbol_data_remote.remote(symbol, start_date, end_date, timeframe)
        for symbol in symbols
    }
    
    # Collect results
    results = {}
    for symbol, future in futures.items():
        try:
            results[symbol] = ray.get(future)
        except Exception as e:
            log.error(f"Failed to load {symbol}: {e}")
            results[symbol] = pd.DataFrame()
    
    return results


def create_ray_dataset(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dataset:
    """
    Create a Ray Dataset for streaming large datasets.
    
    Ray Data handles out-of-core processing, so you can work with
    datasets larger than RAM.
    
    Args:
        symbol: Ticker symbol
        start_date: Start date filter
        end_date: End date filter
        
    Returns:
        Ray Dataset that can be streamed to workers
    """
    parquet_path = settings.data.features_parquet_dir / symbol
    
    if not parquet_path.exists():
        log.warning(f"No data found for symbol: {symbol}")
        return ray.data.from_items([])
    
    # Build file list with date filtering
    parquet_files = []
    for path in parquet_path.glob("dt=*/*.parquet"):
        date_str = path.parent.name.replace("dt=", "")
        
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
            
        parquet_files.append(str(path))
    
    if not parquet_files:
        log.warning(f"No parquet files found for {symbol} in date range")
        return ray.data.from_items([])
    
    # Create Ray Dataset from parquet files
    ds = ray.data.read_parquet(parquet_files)
    
    log.info(f"Created Ray Dataset for {symbol}: {ds.count()} rows")
    return ds


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "close",
    target_transform: str = "log_return",
    lookforward: int = 1,
    test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for training with anti-leakage protections.
    
    Args:
        df: Raw feature DataFrame
        target_col: Column to predict
        target_transform: "log_return", "pct_change", or "raw"
        lookforward: Predict N steps ahead
        test_ratio: Fraction of data for test set
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    df = df.copy()
    
    # Set index
    if "ts" in df.columns:
        df = df.set_index("ts").sort_index()
    
    # Create target
    future_val = df[target_col].shift(-lookforward)
    
    if target_transform == "log_return":
        df["target"] = np.log((future_val + 1e-9) / (df[target_col] + 1e-9))
    elif target_transform == "pct_change":
        df["target"] = (future_val - df[target_col]) / (df[target_col] + 1e-9)
    else:
        df["target"] = future_val
    
    # Drop raw price columns (anti-leakage)
    raw_price_cols = ["open", "high", "low", "close", "vwap"]
    cols_to_drop = [c for c in df.columns if any(
        c == raw or c.startswith(f"{raw}_") for raw in raw_price_cols
    )]
    
    # Metadata columns to drop
    meta_cols = ["target", "symbol", "date", "source", "options", "data_split", target_col]
    drop_cols = list(set(cols_to_drop + meta_cols))
    
    # Get feature columns
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c not in drop_cols]
    
    # Prepare X and y
    X = df[feature_cols].dropna(axis=1, how="all")
    y = df["target"]
    
    # Align and drop NaN targets
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Time-based split
    split_idx = int(len(X) * (1 - test_ratio))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    log.info(f"Prepared data: Train={len(X_train)}, Test={len(X_test)}, Features={len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test

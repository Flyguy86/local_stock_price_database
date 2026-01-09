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
    Load data from Parquet features. 
    Supports multiple tickers via comma-separation (e.g. "GOOGL,VIX").
    First ticker is the PRIMARY (target). Others are merged as features.
    """
    symbols = [s.strip() for s in symbol.split(",")]
    primary_symbol = symbols[0]
    context_symbols = symbols[1:]
    
    # --- Helper to load one symbol ---
    def _load_single(sym):
        path = settings.features_parquet_dir / sym
        if not path.exists():
            raise FileNotFoundError(f"No features data found for {sym}")
        query = f"SELECT * FROM '{path}/**/*.parquet'"
        
        # Handle options filter
        if options_filter:
            safe_filter = options_filter.replace("'", "''")
            query += f" WHERE options = '{safe_filter}'"
        
        # We don't strictly need to order here if we merge on TS later, but good for primary
        query += " ORDER BY ts ASC"
        return duckdb.query(query).to_df()

    # 1. Load Primary
    log.info(f"Loading primary ticker: {primary_symbol}")
    df = _load_single(primary_symbol)
    
    if df.empty:
        raise ValueError(f"No data rows for primary symbol {primary_symbol}")

    # 2. Load and Merge Context Tickers
    for ctx_sym in context_symbols:
        log.info(f"Loading context ticker: {ctx_sym}")
        ctx_df = _load_single(ctx_sym)
        if ctx_df.empty:
            log.warning(f"Context symbol {ctx_sym} has no data. Skipping.")
            continue
            
        # Rename columns to avoid collisions (except ts)
        # We keep 'ts' for the join key
        # We generally drop non-numeric metadata from context to save memory/logic, 
        # but for now we rename e.g. "close" -> "close_VIX"
        cols_to_rename = {c: f"{c}_{ctx_sym}" for c in ctx_df.columns if c != "ts"}
        ctx_df = ctx_df.rename(columns=cols_to_rename)
        
        # Merge via Inner Join to ensure time alignment
        # Any minute missing in EITHER dataset is dropped to ensure data integrity
        df = pd.merge(df, ctx_df, on="ts", how="inner")
        log.info(f"Merged {ctx_sym}. Resulting rows: {len(df)}")

    # Create target: Percent change 'lookforward' steps ahead
    # Simple Example: Predict if price goes up (1) or down (0)
    # Or Predict the next close price. 
    # Let's do simple regression: target is next 'close' price.
    
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in features (Primary Symbol: {primary_symbol})")

    # Shift target backward to align "current features" with "future price"
    df["target"] = df[target_col].shift(-lookforward)
    
    # Drop rows without target (last N rows)
    df = df.dropna(subset=["target"])
    
    # Drop non-numeric for training (except ts)
    # We keep 'ts' for splitting but drop it for model input usually
    return df

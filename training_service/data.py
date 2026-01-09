import pandas as pd
import numpy as np
from pathlib import Path
from .config import settings
import logging
import duckdb

log = logging.getLogger("training.data")

def get_feature_map() -> dict[str, list[str]]:
    """
    Scans all feature parquet files to build a map of:
    Options Config -> List[Symbols]
    """
    base_path = settings.features_parquet_dir
    if not base_path.exists():
        return {}
    
    glob_pattern = str(base_path / "**/*.parquet")
    
    # We use regex to extract the symbol from the path.
    # Path structure: .../features_parquet/SYMBOL/dt=...
    # We assume 'features_parquet/' is part of the path one level above symbols.
    
    sql = f"""
    SELECT 
        options,
        array_agg(DISTINCT regexp_extract(filename, 'features_parquet/([^/]+)/', 1)) as symbols
    FROM read_parquet('{glob_pattern}', filename=true)
    GROUP BY options
    """
    
    try:
        res = duckdb.query(sql).fetchall()
        mapping = {}
        for row in res:
            opt = row[0]
            syms = row[1]
            if opt:
                valid_syms = sorted([s for s in syms if s])
                mapping[opt] = valid_syms
        return mapping
    except Exception as e:
        log.error(f"Failed to scan feature map: {e}")
        return {}

def get_data_options(symbol: str = None) -> list[str]:
    """
    Get distinct 'options' strings from parquet files.
    """
    # Use the map to ensure consistency
    mapping = get_feature_map()
    
    if symbol:
        # Find options where this symbol exists
        return sorted([k for k, v in mapping.items() if symbol in v])
    else:
        return sorted(list(mapping.keys()))

def load_training_data(symbol: str, target_col: str = "close", lookforward: int = 1, options_filter: str = None, timeframe: str = "1m") -> pd.DataFrame:
    """
    Load data from Parquet features. 
    Supports multiple tickers via comma-separation (e.g. "GOOGL,VIX").
    First ticker is the PRIMARY (target). Others are merged as features.
    
    timeframe: Resample interval (e.g. "1m" (check if null), "10m", "1h", "4h", "8h").
               Default "1m" means no resampling (assuming source is 1m).
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

    # 3. Resample if needed
    if timeframe and timeframe != "1m":
        log.info(f"Resampling data to {timeframe}")
        df = df.set_index("ts").sort_index()
        
        # Build aggregation dictionary
        agg_dict = {}
        split_col_found = None
        
        for col in df.columns:
             # Identify split column name (usually 'data_split')
            is_split_col = (col == "data_split") or (col.endswith("_split"))
            if is_split_col:
                # Custom aggregator: If ANY record in this bucket is 'test', the whole bucket is 'test'
                # This is conservative: avoids leaking test data into a 'train' bucket
                def aggressive_test_label(series):
                    vals = set(series.astype(str).unique())
                    if "test" in vals: return "test"
                    return "train" # fallback
                agg_dict[col] = aggressive_test_label
                split_col_found = col
            elif col == "open": agg_dict[col] = "first"
            elif col == "high": agg_dict[col] = "max"
            elif col == "low": agg_dict[col] = "min"
            elif col == "close": agg_dict[col] = "last"
            elif col == "volume": agg_dict[col] = "sum"
            elif col == "trade_count": agg_dict[col] = "sum"
            elif col == "vwap": agg_dict[col] = "mean" # approximate
            else:
                agg_dict[col] = "last"
        
        df = df.resample(timeframe).agg(agg_dict)
        # Drop rows where main cols are NaN (e.g. gaps in trading days)
        df = df.dropna(subset=["close"])
        df = df.reset_index()
        log.info(f"Resampled rows: {len(df)}")

    # Create target: Percent change 'lookforward' steps ahead
    # Simple Example: Predict if price goes up (1) or down (0)
    # Or Predict the next close price. 
    # Let's do simple regression: target is next 'close' price.
    
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in features (Primary Symbol: {primary_symbol})")

    # Shift target backward to align "current features" with "future price"
    df["target"] = df[target_col].shift(-lookforward)
    
    # --- PREVENT DATA LEAKAGE ---
    # If using segmentation, we must handle the boundary where Train -> Test.
    # At Row T (Train), the target comes from Row T+1. If T+1 is 'test', then training on Row T
    # uses 'test' data as a label. This is a leak.
    
    # 1. Detect split column again (if not already found during resample loop)
    split_col = None
    for c in df.columns:
        if c == 'data_split' or c.endswith('_split'):
            split_col = c
            break
            
    if split_col:
        # Check source of future target
        future_split = df[split_col].shift(-lookforward)
        
        # Identify leak rows: Current is Train, Future is Test
        # We assume 'train' and 'test' are the string values.
        is_leak = (df[split_col] == 'train') & (future_split == 'test')
        leak_count = is_leak.sum()
        
        if leak_count > 0:
            log.warning(f"Dropping {leak_count} rows to prevent Train->Test leakage (Lookforward={lookforward})")
            df = df[~is_leak] # Drop them

    # Drop rows without target (last N rows)
    df = df.dropna(subset=["target"])
    
    return df

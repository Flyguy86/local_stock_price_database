import pandas as pd
import numpy as np
from pathlib import Path
from .config import settings
import logging
import duckdb

log = logging.getLogger("training.data")

import time

def get_feature_map() -> dict[str, list[str]]:
    """
    Scans all feature parquet files to build a map of:
    Options Config -> List[Symbols]
    """
    base_path = settings.features_parquet_dir
    if not base_path.exists():
        log.warning(f"Feature path does not exist: {base_path}")
        return {}
    
    glob_pattern = str(base_path / "**/*.parquet")
    log.info(f"Scanning feature map with pattern: {glob_pattern}")
    
    start_ts = time.time()
    
    # We use regex to extract the symbol from the path.
    # Path structure: .../features_parquet/SYMBOL/dt=...
    # We assume 'features_parquet/' is part of the path one level above symbols.
    
    sql = f"""
    SELECT 
        options,
        array_agg(DISTINCT regexp_extract(filename, 'features_parquet/([^/]+)/', 1)) as symbols
    FROM read_parquet('{glob_pattern}', filename=true, union_by_name=true)
    GROUP BY options
    """
    
    try:
        res = duckdb.query(sql).fetchall()
        duration = time.time() - start_ts
        log.info(f"Feature map scan completed in {duration:.2f}s. Found {len(res)} raw groups.")
        
        mapping = {}
        for row in res:
            opt = row[0]
            syms = row[1]
            
            # Detailed Logging
            log.info(f"Scan Group: '{opt}' (Type: {type(opt)}) | Symbols: {len(syms)}")
            
            if not opt:
                opt = '{"desc": "Legacy / No Config"}' # Fallback for old data
            
            valid_syms = sorted([s for s in syms if s])
            mapping[opt] = valid_syms

        if not mapping:
             log.warning("Scan returned 0 valid mappings. Check if Parquet files contain 'options' column.")
             
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

def load_training_data(symbol: str, target_col: str = "close", lookforward: int = 1, options_filter: str = None, timeframe: str = "1m", target_transform: str = "none") -> pd.DataFrame:
    """
    Load data from Parquet features. 
    Supports multiple tickers via comma-separation.
    target_transform: 'none', 'log_return', 'pct_change'
    """
    symbols = [s.strip() for s in symbol.split(",")]
    primary_symbol = symbols[0]
    context_symbols = symbols[1:]
    
    # --- Helper to load one symbol ---
    def _load_single(sym):
        path = settings.features_parquet_dir / sym
        if not path.exists():
            raise FileNotFoundError(f"No features data found for {sym}")
        
        # We use strict union_by_name to handle files with/without 'options' column gracefully
        query = f"SELECT * FROM read_parquet('{path}/**/*.parquet', union_by_name=true)"
        
        # Handle options filter
        if options_filter:
            # Handle Legacy key
            if 'Legacy / No Config' in options_filter:
                 log.info(f"Applying legacy filter for {sym}")
                 # Match NULL (missing column) or Empty string
                 query += " WHERE options IS NULL OR options = ''"
            # Robust handling for empty options
            elif options_filter.strip() in ["{}", ""]:
                 log.info(f"Applying flexible empty option filter for '{options_filter}'")
                 query += " WHERE options = '{}' OR options = '' OR options IS NULL"
            else:
                 safe_filter = options_filter.replace("'", "''")
                 query += f" WHERE options = '{safe_filter}'"
        
        # We don't strictly need to order here if we merge on TS later, but good for primary
        query += " ORDER BY ts ASC"
        return duckdb.query(query).to_df()

    # 1. Load Primary
    log.info(f"Loading primary ticker: {primary_symbol} from {settings.features_parquet_dir / primary_symbol}")
    log.info(f"Options filter: '{options_filter}' (Type: {type(options_filter)})")
    
    try:
        df = _load_single(primary_symbol)
        log.info(f"Loaded {len(df)} rows for {primary_symbol}")
    except Exception as e:
        log.error(f"Error loading {primary_symbol}: {e}")
        raise
    
    if df.empty:
        # Fallback debug: Check if ANY data exists ignoring options
        base_df = duckdb.query(f"SELECT count(*) as c, options FROM '{settings.features_parquet_dir / primary_symbol}/**/*.parquet' GROUP BY options").to_df()
        log.error(f"DEBUG: Found data for {primary_symbol} with these options: {base_df.to_dict(orient='records')}")
        raise ValueError(f"No data rows for primary symbol {primary_symbol}. Filter='{options_filter}'")

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
                    # Priority: Test > Train. If bucket contains any 'test' data, label it 'test'
                    # so it gets excluded from training (or handled as test boundary).
                    if "test" in vals: return "test"
                    return "train" # fallback
                
                agg_dict[col] = aggressive_test_label
                split_col_found = col
                log.info(f"Leakage Protection: Applied aggressive split aggregation to '{col}'")

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

    # Target Generation
    future_val = df[target_col].shift(-lookforward)
    
    if target_transform == "log_return":
        # Log Return = ln(Future / Current)
        # Add small epsilon to avoid div/0 if accidentally 0 (though prices shouldn't be)
        df["target"] = np.log((future_val + 1e-9) / (df[target_col] + 1e-9))
        log.info(f"Target: Log Return of {target_col} (lookforward={lookforward})")
    elif target_transform == "pct_change":
        # Pct Change = (Future - Current) / Current
        df["target"] = (future_val - df[target_col]) / (df[target_col] + 1e-9)
        log.info(f"Target: Pct Change of {target_col} (lookforward={lookforward})")
    else:
        # Raw Value (Non-Stationary trap!)
        df["target"] = future_val
        log.warning(f"Target: PREDICTING RAW {target_col}. Be wary of stationarity issues!")
    
    # --- PREVENT DATA LEAKAGE ---
    # If using segmentation, we must handle the boundary where Train -> Test.
    # At Row T (Train), the target comes from Row T+1. If T+1 is 'test', then training on Row T
    # uses 'test' data as a label. This is a leak.
    
    # Key concept: 1m -> 1h resampling has ALREADY conservatively marked mixed buckets as 'test'.
    # Now we check the boundary between the last pure 'train' bucket and the first 'test' bucket.

    # 1. Detect split column again (if not already found during resample loop)
    split_col = None
    for c in df.columns:
        if c == 'data_split' or c.endswith('_split'):
            split_col = c
            break
            
    if split_col:
        # Check source of future target
        # If lookforward=1, Row[i] uses Row[i+1][target_col]
        # We perform check: If Row[i].split == 'train' AND Row[i+1].split == 'test', DROP Row[i].
        future_split = df[split_col].shift(-lookforward)
        
        # Identify leak rows: Current is Train, Future is Test
        # We assume 'train' and 'test' are the string values.
        is_leak = (df[split_col] == 'train') & (future_split == 'test')
        leak_count = is_leak.sum()
        
        if leak_count > 0:
            log.warning(f"Leakage Protection: Dropping {leak_count} rows at Train->Test boundary (Lookforward={lookforward})")
            df = df[~is_leak] # Drop them

    # Drop rows without target (last N rows)
    df = df.dropna(subset=["target"])
    
    return df

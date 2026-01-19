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
    
    # Strip reference_symbols from options_filter since it's not in parquet data
    parquet_options_filter = options_filter
    if options_filter:
        try:
            import json
            opts = json.loads(options_filter)
            if 'reference_symbols' in opts:
                del opts['reference_symbols']
                parquet_options_filter = json.dumps(opts, sort_keys=True)
                log.info(f"Stripped reference_symbols, using filter: {parquet_options_filter}")
        except (json.JSONDecodeError, TypeError):
            pass  # Not JSON, use as-is
    
    # --- Helper to load one symbol ---
    def _load_single(sym):
        path = settings.features_parquet_dir / sym
        if not path.exists():
            raise FileNotFoundError(f"No features data found for {sym}")
        
        # We use strict union_by_name to handle files with/without 'options' column gracefully
        query = f"SELECT * FROM read_parquet('{path}/**/*.parquet', union_by_name=true)"
        
        # Handle options filter (using parquet_options_filter without reference_symbols)
        if parquet_options_filter:
            # Handle Legacy key
            if 'Legacy / No Config' in parquet_options_filter:
                 log.info(f"Applying legacy filter for {sym}")
                 # Match NULL (missing column) or Empty string
                 query += " WHERE options IS NULL OR options = ''"
            # Robust handling for empty options
            elif parquet_options_filter.strip() in ["{}", ""]:
                 log.info(f"Applying flexible empty option filter for '{parquet_options_filter}'")
                 query += " WHERE options = '{}' OR options = '' OR options IS NULL"
            else:
                 safe_filter = parquet_options_filter.replace("'", "''")
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
        
        # Validate merge didn't introduce excessive NaNs
        nan_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) if df.shape[0] > 0 else 0
        if nan_pct > 0.10:
            log.warning(f"After merging {ctx_sym}: {nan_pct*100:.2f}% NaN values (high)")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            log.warning(f"After merging {ctx_sym}: {inf_count} infinite values detected, replacing with NaN")
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

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
    elif target_transform == "log":
        # Log Price: predict log(Future Price)
        # Useful for regression when price levels matter
        df["target"] = np.log(future_val + 1e-9)
        log.info(f"Target: Log of {target_col} (lookforward={lookforward})")
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

    # --- DROP RAW PRICE COLUMNS TO PREVENT LEAKAGE / OVERFITTING ---
    # We only want to train on derived stationary features (returns, techncials), 
    # not the raw price level itself.
    raw_price_cols = ["open", "high", "low", "close", "vwap"]
    
    # Also drop context raw prices (e.g. open_MSFT, close_QQQ)
    cols_to_drop = []
    for col in df.columns:
        # Check pure raw price 
        if col in raw_price_cols:
            cols_to_drop.append(col)
            continue
            
        # Check context raw price (e.g. "close_MSFT")
        for raw in raw_price_cols:
            if col.startswith(f"{raw}_"):
                cols_to_drop.append(col)
                break
                
    if cols_to_drop:
        # Only drop if they are NOT the target column (rare edge case where target is raw close)
        # If target_col is 'close', and we are predicting 'close' (raw), we actually NEED 'close' to lag safely.
        # But usually we predict 'target' which is shifted. 
        # The 'df' returned here is fed to X. X excludes 'target'. 
        # So we can safely drop these from X's potential candidates.
        
        # HOWEVER: Use caution. Some indicators like 'dist_vwap' rely on them? 
        # No, 'dist_vwap' is already calculated in feature_service.
        
        # We just drop them from the dataframe so trainer doesn't see them.
        # Ensure we don't drop columns needed for logic later (like target_col for classifier direction)
        
        # We will keep them for now, but rely on Trainer to drop them? 
        # No, better to drop here before Trainer sees them as features.
        
        # Exception: We often need 'close' or 'open' to calculate the final PnL, graph, or Price RMSE.
        # So we keep the primary 'target_col' (e.g. 'close') for reference, but MUST ensure it's dropped from X in trainer.
        
        # FIX: The previous logic relied on `target_col` string check ("close" vs "close_MSFT"), which was buggy.
        # We need to strictly DROP all raw price columns from ALL tickers provided.
        # We RE-ADD the primary target_col solely for bookkeeping if it was dropped.
        
        # 1. Identify what we want to KEEP for bookkeeping (the primary target raw column)
        keep_col = target_col if target_col in df.columns else None
        
        # 2. Perform Drops
        final_drop = [c for c in cols_to_drop if c not in ["target", "ts", "data_split"]]
        # If the keep_col is in the drop list, we still drop it from features, 
        # BUT we handle it carefully: Trainer splits X (features) and y (target) and meta.
        # The issue is: If 'close' is in X, the model uses it.
        # We must DROP it from X in Trainer. Here in Data Loader, we can keep it in the DF *if* we ensure Trainer strips it.
        
        # The user report says "close" leaked.
        # This implies Trainer did NOT strip 'close' from X, or Data Loader left context closes (close_MSFT, close_QQQ).
        
        # The logic below says: drop everything in cols_to_drop EXCEPT target_col.
        # If target_col is 'close', then 'close' remains in the DF. 
        # Then in Trainer, 'close' MUST be in `drop_cols`.
        
        # Let's verify context columns. "close_QQQ" -> starts with "close_".
        # If primary is "close", keep_col="close".
        # "close_QQQ" is not "close", so it is dropped. Correct.
        
        # Wait, the user report lists "macd_line_QQQ", "return_z_score_20_MSFT"... 
        # It does NOT list "close" or "close_QQQ" in the Feature Analysis content provided above.
        # The top features are MACD, Z-Score, Vol Mom. 
        # So where is the leak?
        
        # User says: "looks like we have data leaks on 'Close'".
        # But features used count is 58.
        # The list shows: macd_line_QQQ, macd_signal_QQQ... are these derived from FUTURE?
        # NO, MACD is lagging.
        
        # Look at the MSE: 0.00000.  RMSE: 0.00104.
        # If target is log_return, typical values are 0.0001 to 0.005. 
        # An RMSE of 0.00104 is heavily correlated (R2 ~ 80%?), but maybe not 100% leak.
        # 0.00000 MSE might just be display rounding?
        
        # Let's look at `log_return_1m_QQQ` (Rank 12).
        # If we predict `log_return_1m` (Primary), and we include `log_return_1m_QQQ` (Context),
        # And QQQ moves identical to target (SPY?), then it's highly correlated.
        # That is NOT leakage (using future data), that is just high Correlation (Multicollinearity).
        
        # HOWEVER: If `log_return_1m` uses (Close_T / Close_T-1), and we predict (Close_T+1 / Close_T).
        # No leak there.
        
        # Wait, did we enable 'Shift'? 
        # In `pipeline.py`: out["log_return_1m"] = np.log(out["close"] / out["close"].shift(1))
        # At time T, `log_return_1m` describes T vs T-1.
        # Target at T (lookforward=1) describes T+1 vs T.
        # So Features=Past, Target=Future.
        
        # Is it possible `timeframe="1m"` resampling is broken?
        # If we resample 1m -> 1m, the loop `for col in df.columns` inside `resample` block is SKIPPED?
        # Lines 150+: `if timeframe and timeframe != "1m":`
        # So for 1m, no Aggregation logic runs. 
        
        # BUT: The anti-leakage drop logic is lines ~250+. This runs for 1m too.
        
        # Let's make sure context close prices are definitely dropped.
        # The user report doesn't show "close_QQQ" in the list. 
        # It shows `return_1m_QQQ`.
        # This confirms Raw Prices ARE gone.
        
        # Re-reading Report: "Mean Squared Error 0.00000", "RMSE 0.00104". 
        # If MSE is 1.08e-6, it displays as 0.00000.
        # RMSE 0.001 is 0.1%. 
        # For a stock, 0.1% error on 1-min return is actually decent/realistic if volatility is low.
        # Maybe it's NOT a leak, but the user THINKS it is because MSE shows 0.00000?
        
        # But wait, `macd_line_QQQ` is #1 feature. 
        
        # CRITICAL: `return_1m` vs Target.
        # If we predict T+1 return.
        # And we feed T return.
        # Momentum? 
        
        # User said: "looks like we have data leaks on Close".
        # Maybe they mean the target variable `close` was included?
        # In Trainer, we drop `drop_cols = [..., target_col]`.
        # I modified Trainer to look for `target_col` in drop_cols.
        # Let's double check Trainer again.
        
        cols_to_drop = [c for c in cols_to_drop if c not in ["target", "ts", "data_split", target_col]]
        log.info(f"Anti-Leakage: Dropping {len(cols_to_drop)} raw price columns: {cols_to_drop[:5]}...")
        df = df.drop(columns=cols_to_drop)

    # Drop rows without target (last N rows)
    df = df.dropna(subset=["target"])
    
    # 5. Final Validation before returning
    final_nan_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) if df.shape[0] > 0 else 0
    log.info(f"Final data quality: {len(df)} rows, {len(df.columns)} columns, {final_nan_pct*100:.2f}% NaN")
    
    if final_nan_pct > 0.15:
        log.error(f"CRITICAL: Final dataset has {final_nan_pct*100:.2f}% NaN values (threshold: 15%)")
        # List columns with high NaN
        high_nan_cols = df.columns[df.isna().sum() > len(df) * 0.2].tolist()
        if high_nan_cols:
            log.error(f"Columns with >20% NaN: {high_nan_cols}")
        raise ValueError(f"Dataset quality check failed: {final_nan_pct*100:.2f}% NaN exceeds 15% threshold")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        log.warning(f"Final dataset has {inf_count} infinite values, replacing with NaN")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    log.info("✅ Data validation passed")
    return df

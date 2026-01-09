import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import duckdb

log = logging.getLogger("simulation.core")

# Settings / Config from environment or defaults
MODELS_DIR = Path("/app/data/models")
FEATURES_PATH = Path("/app/data/features_parquet")
METADATA_DB_PATH = Path("/app/data/duckdb/models.db")

def get_available_models():
    """Lists available trained models (.joblib files) with metadata."""
    if not MODELS_DIR.exists():
        return []
    
    # 1. Get physical files
    model_files = {}
    for p in MODELS_DIR.glob("*.joblib"):
        model_files[p.stem] = {"path": str(p), "filename": p.name}
        
    if not model_files:
        return []

    # 2. Get Metadata from DB
    metadata_map = {}
    if METADATA_DB_PATH.exists():
        try:
            # Connect read-only to avoid locks
            with duckdb.connect(str(METADATA_DB_PATH), read_only=True) as conn:
                # Check if table exists
                tables = conn.execute("SHOW TABLES").fetchall()
                if any(t[0] == 'models' for t in tables):
                    # Fetch extra columns: timeframe, data_options
                    # We might need to handle schemas where columns don't exist yet via try/except or rigorous selecting
                    # Using SELECT * is risky if schema changed. Explicit select is better.
                    # We try to select all including new columns.
                    try:
                        rows = conn.execute("SELECT id, algorithm, symbol, created_at, metrics, timeframe, data_options FROM models").fetchall()
                        for r in rows:
                            mid, algo, sym, created, metrics, tf, d_opt = r
                            metadata_map[mid] = {
                                "algorithm": algo,
                                "symbol": sym, 
                                "created_at": created,
                                "metrics": metrics,
                                "timeframe": tf,
                                "data_options": d_opt
                            }
                    except Exception as e:
                         log.warning(f"Error reading extended metadata (using fallback): {e}")
                         # Fallback for old schema
                         rows = conn.execute("SELECT id, algorithm, symbol, created_at, metrics FROM models").fetchall()
                         for r in rows:
                            mid, algo, sym, created, metrics = r
                            metadata_map[mid] = {
                                "algorithm": algo,
                                "symbol": sym,
                                "created_at": created,
                                "metrics": metrics,
                                "timeframe": "1m",
                                "data_options": None
                            }
                        
        except Exception as e:
            log.warning(f"Failed to read metadata DB: {e}")

    # 3. Merge
    result = []
    for mid, info in model_files.items():
        meta = metadata_map.get(mid, {})
        
        # Build a nice display name
        if meta:
            # Format: Symbol(TF) - Algo - Date
            # e.g. "AAPL(1h) - RandomForest - 2023-10-27"
            date_str = str(meta.get("created_at", ""))[:10]
            tf = meta.get("timeframe", "1m")
            sym = meta.get("symbol", "?")
            display = f"{sym} ({tf}) | {meta.get('algorithm', 'Unknown')} | {date_str}"
        else:
            display = f"Unknown Model ({mid[:8]})"
            
        result.append({
            "id": mid, # Note: mid is filename stem (uuid)
            "name": display,
            "path": info["path"],
            "metadata": meta
        })
        
    # Sort by creation date if available (or name)
    result.sort(key=lambda x: x.get("metadata", {}).get("created_at", "") or "", reverse=True)
    
    return result

def get_available_tickers():
    """Lists available tickers based on feature directories."""
    if not FEATURES_PATH.exists():
        return []
    
    tickers = []
    for p in FEATURES_PATH.iterdir():
        if p.is_dir():
            tickers.append(p.name)
    return sorted(tickers)

def load_simulation_data(symbol_str: str, timeframe: str = "1m", options_filter: str = None) -> pd.DataFrame:
    """
    Loads feature data for simulation, replicating logic from training_service/data.py
    Supports multi-symbol (comma separated), context merging, and resampling.
    """
    symbols = [s.strip() for s in symbol_str.split(",")]
    primary_symbol = symbols[0]
    context_symbols = symbols[1:]
    
    # --- Helper to load one symbol ---
    def _load_single(sym):
        path = FEATURES_PATH / sym
        if not path.exists():
            # In simulation, we might not want to crash if context is missing, but for strictness let's warn and return empty
            log.warning(f"No features data found for {sym}")
            return pd.DataFrame()
            
        print(f"Loading {sym} from {path}...")
        query = f"SELECT * FROM '{path}/**/*.parquet'"
        if options_filter:
            safe_filter = options_filter.replace("'", "''")
            query += f" WHERE options = '{safe_filter}'"
        
        query += " ORDER BY ts ASC"
        try:
            return duckdb.query(query).to_df()
        except Exception as e:
            log.error(f"Error loading {sym}: {e}")
            return pd.DataFrame()

    # 1. Load Primary
    df = _load_single(primary_symbol)
    if df.empty:
        raise ValueError(f"No data for primary symbol {primary_symbol}")

    # 2. Load and Merge Context
    for ctx_sym in context_symbols:
        ctx_df = _load_single(ctx_sym)
        if ctx_df.empty:
            continue
            
        cols_to_rename = {c: f"{c}_{ctx_sym}" for c in ctx_df.columns if c != "ts"}
        ctx_df = ctx_df.rename(columns=cols_to_rename)
        
        df = pd.merge(df, ctx_df, on="ts", how="inner")
        
    # 3. Resample
    if timeframe and timeframe != "1m":
        df = df.set_index("ts").sort_index()
        agg_dict = {}
        for col in df.columns:
            # Custom aggregator: If ANY record in this bucket is 'test', the whole bucket is 'test'
            if "split" in col or "data_split" in col:
                 # Conservative approach for simulation too?
                 # Actually for simulation we treat all data as 'historical input'
                 # But we might want to respect the split labels for the chart visualization
                 def aggressive_test_label(series):
                    vals = set(series.astype(str).unique())
                    if "test" in vals: return "test"
                    return "train"
                 agg_dict[col] = aggressive_test_label
            elif "open" in col.lower(): agg_dict[col] = "first"
            elif "high" in col.lower(): agg_dict[col] = "max"
            elif "low" in col.lower(): agg_dict[col] = "min"
            elif "close" in col.lower(): agg_dict[col] = "last"
            elif "volume" in col.lower(): agg_dict[col] = "sum"
            elif "trade_count" in col.lower(): agg_dict[col] = "sum"
            elif "vwap" in col.lower(): agg_dict[col] = "mean"
            else:
                agg_dict[col] = "last"
                
        df = df.resample(timeframe).agg(agg_dict)
        # Drop gaps
        df = df.dropna(subset=[c for c in df.columns if "close" in c][:1]) # Check primary close
        df = df.reset_index()

    return df

def run_simulation(model_id: str, ticker: str, initial_cash: float):
    """
    Runs a backtest simulation.
    Note: 'ticker' param passed from UI usually overrides model symbol? 
    Actually, usually we simulate a model on valid data.
    If the model was trained on GOOGL+VIX, we MUST provide GOOGL+VIX features.
    
    If the user passed a custom 'ticker' override in the UI (e.g. testing a GOOGL model on MSFT), 
    we need to be careful.
    
    Strategy:
    1. Load Model Metadata to get training config (timeframe, context symbols).
    2. If user provided a generic ticker (e.g. "MSFT") but model expects "GOOGL,VIX", 
       we assume the user wants to substitute the PRIMARY symbol.
       So we construct "MSFT,VIX".
    """
    model_path = MODELS_DIR / model_id
    if not model_path.exists():
        if not model_id.endswith(".joblib"):
             model_path = MODELS_DIR / f"{model_id}.joblib"

    if not model_path.exists():
        raise ValueError(f"Model not found: {model_id}")
        
    # Get Metadata
    metadata = {}
    if METADATA_DB_PATH.exists():
         with duckdb.connect(str(METADATA_DB_PATH), read_only=True) as conn:
             row = conn.execute("SELECT symbol, timeframe, data_options FROM models WHERE id = ?", [model_id.replace(".joblib", "")]).fetchone()
             if row:
                 metadata = {"symbol": row[0], "timeframe": row[1], "data_options": row[2]}

    trained_symbol_str = metadata.get("symbol", ticker) # Fallback to passed ticker if metadata missing
    timeframe = metadata.get("timeframe", "1m")
    data_options = metadata.get("data_options")
    
    # Handle Symbol Substitution for Cross-Ticker Testing
    # User selected 'ticker' in Simulation UI. Model was trained on 'trained_symbol_str'.
    # If they differ, we assume substitution for the PRIMARY ticker only.
    # Ex: Model="GOOGL,VIX", User="MSFT". We load "MSFT,VIX".
    
    trained_primary = trained_symbol_str.split(",")[0].strip()
    trained_context = trained_symbol_str.split(",")[1:]
    
    target_load_str = ticker # Start with user request
    if trained_context:
        # Append context from training config
        target_load_str = ",".join([ticker] + [s.strip() for s in trained_context])
        
    log.info(f"Loading model form {model_path}")
    model = joblib.load(model_path)
    
    log.info(f"Loading data for {target_load_str} (TF={timeframe}, Opts={data_options})")
    df = load_simulation_data(target_load_str, timeframe, data_options)
    
    if df.empty:
        raise ValueError(f"No data for {target_load_str}")

    # Prepare features for the model.
    drop_cols = ["target", "ts", "symbol", "date", "source", "options", "target_col_shifted", "dt"]
    # Also drop split columns if any (train/test markers)
    for col in df.columns:
        if "train" in col.lower() or "test" in col.lower():
            # If it's a categorical column
            if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
                 drop_cols.append(col)

    # Filter numeric
    df_numeric = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in df_numeric.columns if c not in drop_cols]
    
    # Check if we have features
    if not feature_cols:
        raise ValueError("No feature columns found")

    X = df[feature_cols].copy()
    
    # Attempt to align features with model expectations
    if hasattr(model, "feature_names_in_"):
        required_features = list(model.feature_names_in_)
        # Check missing
        missing = [c for c in required_features if c not in X.columns]
        if missing:
             # Try to find them in original df (maybe we dropped them too early?)
             for m in missing:
                 if m in df.columns:
                     X[m] = df[m]
        
        # Check again
        missing = [c for c in required_features if c not in X.columns]
        if missing:
            raise ValueError(f"Model requires features not present in data: {missing}")
        
        # Reorder and filter to match model
        X = X[required_features]
    
    # Handle NaNs - Fill with 0 or drop? Model might handle it?
    # SimpleImputer was used in training potentially?
    # For simulation, we'll iterate or batch predict.
    # To keep it simple, drop rows with NaNs in features
    X = X.fillna(0) # Simple fill for now to ensure run
    
    # Predict
    try:
        # Check if model has predict_proba (Classifier) or predict (Regressor)
        # The user mentioned "Buy / Sell indicators"
        # If classifier: 1 = Buy (Up), 0 = Sell/Hold (Down)
        preds = model.predict(X)
        
        # If it's a regression model, we might interpret positive return as Buy?
        # But let's assume binary for now or generic signal.
        # existing trainer handles both.
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        # Try adjusting columns if mismatch
        # Sometimes feature order matters or extra columns exist
        # If model expects specific features, we might need metadata.
        # Fallback: try to warn/fail
        raise e

    # Add predictions to DF
    df_sim = df.copy()
    # Align indices if we dropped rows (we didn't yet, X matches df rows count here)
    df_sim["prediction"] = preds
    
    # Generate Signals
    # Logic:
    # If Classifier: 1 = Buy, 0 = Sell
    # If Regressor: Prediction > Threshold (e.g. 0) = Buy, else Sell
    
    # Let's inspect prediction values slightly
    is_classifier = hasattr(model, "predict_proba") or len(np.unique(preds)) <= 2
    
    if is_classifier:
        df_sim["signal"] = df_sim["prediction"].apply(lambda x: 1 if x == 1 else 0)
    else:
        # Regressor (predicting future price or return?)
        # For now assume predicting return: > 0 means Buy
        df_sim["signal"] = df_sim["prediction"].apply(lambda x: 1 if x > 0 else 0)

    # Walk forward simulation
    cash = initial_cash
    shares = 0
    portfolio_values = []
    trades = []
    last_buy_price = 0.0
    
    # Benchmark: Buy and Hold
    # Start with all cash buying shares at first close
    initial_price = df_sim.iloc[0]["close"]
    benchmark_shares = initial_cash / initial_price
    
    for idx, row in df_sim.iterrows():
        price = row["close"]
        ts = row["ts"]
        signal = row["signal"]
        
        # Execute Strategy Logic (Simple: Buy on 1, Sell on 0)
        # If signal 1 and no shares: Buy Max
        # If signal 0 and shares: Sell All
        
        action = None
        if signal == 1 and shares == 0:
            # Whole shares only
            possible_shares = int(cash // price)
            if possible_shares > 0:
                shares = possible_shares
                cost = shares * price
                cash -= cost
                last_buy_price = price
                action = "BUY"
                trades.append({
                    "ts": ts,
                    "type": "BUY",
                    "price": price,
                    "shares": shares,
                    "value": cost,
                    "pnl": 0.0,
                    "pnl_pct": 0.0
                })
        elif signal == 0 and shares > 0:
            proceeds = shares * price
            
            # Calculate PnL
            pnl = proceeds - (shares * last_buy_price)
            pnl_pct = (price - last_buy_price) / last_buy_price * 100
            
            cash += proceeds
            shares = 0
            action = "SELL"
            trades.append({
                "ts": ts,
                "type": "SELL",
                "price": price,
                "shares": shares, # 0
                "value": proceeds, # value of the sale
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })
            
        current_val = cash + (shares * price)
        portfolio_values.append(current_val)
    
    df_sim["strategy_equity"] = portfolio_values
    df_sim["benchmark_equity"] = df_sim["close"] * benchmark_shares
    
    # Results
    stats = {
        "start_date": df_sim["ts"].min().isoformat(),
        "end_date": df_sim["ts"].max().isoformat(),
        "final_strategy_value": df_sim["strategy_equity"].iloc[-1],
        "final_benchmark_value": df_sim["benchmark_equity"].iloc[-1],
        "strategy_return_pct": (df_sim["strategy_equity"].iloc[-1] - initial_cash) / initial_cash * 100,
        "benchmark_return_pct": (df_sim["benchmark_equity"].iloc[-1] - initial_cash) / initial_cash * 100,
        "total_trades": len(trades)
    }

    # Chart Data (Downsample if needed?)
    chart_data = df_sim[["ts", "strategy_equity", "benchmark_equity"]].copy()
    chart_data["ts"] = chart_data["ts"].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        "stats": stats,
        "trades": trades,
        "chart_data": chart_data.to_dict(orient="records")
    }

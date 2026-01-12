import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import duckdb
import uuid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# Settings / Config from environment or defaults
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/app/data")
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / "data"

MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(DATA_DIR / "models")))
BOTS_DIR = MODELS_DIR / "bots"
FEATURES_PATH = Path(os.environ.get("FEATURES_PATH", str(DATA_DIR / "features_parquet")))
METADATA_DB_PATH = DATA_DIR / "duckdb/models.db"

log = logging.getLogger("simulation.core")

def ensure_sim_history_table():
    """Ensures the simulation history table exists in DuckDB."""
    if not METADATA_DB_PATH.parent.exists():
        METADATA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
    try:
        with duckdb.connect(str(METADATA_DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulation_history (
                    id VARCHAR PRIMARY KEY,
                    timestamp VARCHAR,
                    model_id VARCHAR,
                    ticker VARCHAR,
                    return_pct DOUBLE,
                    trades_count INTEGER,
                    hit_rate DOUBLE,
                    sqn DOUBLE,
                    params JSON
                )
            """)
            
            # Simple migration attempt for older schemas
            try:
                conn.execute("ALTER TABLE simulation_history ADD COLUMN hit_rate DOUBLE")
            except Exception:
                pass # Already exists or other error

            try:
                conn.execute("ALTER TABLE simulation_history ADD COLUMN sqn DOUBLE")
            except Exception:
                pass 
                
            try:
                conn.execute("ALTER TABLE simulation_history ADD COLUMN params JSON")
            except Exception:
                pass 
                
    except Exception as e:
        log.error(f"Failed to ensure sim_history table: {e}")

def save_simulation_history(model_id, ticker, stats, params):
    """Saves a simulation run result to DB."""
    try:
        ensure_sim_history_table()
        
        record_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        
        # params dict to json string
        params_json = json.dumps(params)
        
        with duckdb.connect(str(METADATA_DB_PATH)) as conn:
            conn.execute("""
                INSERT INTO simulation_history 
                (id, timestamp, model_id, ticker, return_pct, trades_count, hit_rate, sqn, params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record_id, 
                ts, 
                model_id, 
                ticker, 
                stats.get('strategy_return_pct', 0.0),
                stats.get('total_trades', 0),
                stats.get('hit_rate_pct', 0.0),
                stats.get('sqn', 0.0),
                params_json
            ])
            log.info(f"Saved simulation history: {record_id}")
            
    except Exception as e:
        log.error(f"Failed to save simulation history: {e}")

def get_simulation_history(limit=50):
    """Retrieves recent simulation history."""
    try:
        ensure_sim_history_table()
        with duckdb.connect(str(METADATA_DB_PATH), read_only=True) as conn:
             # Check if table exists (it might not if valid but empty)
             tables = conn.execute("SHOW TABLES").fetchall()
             if not any(t[0] == 'simulation_history' for t in tables):
                 return []
                 
             # Handle schema evolution gracefully
             columns = "id, timestamp, model_id, ticker, return_pct, trades_count, hit_rate, sqn, params"
             # If columns missing in old valid DB, this might fail, but ensure_table tries to add them.
             
             rows = conn.execute(f"""
                SELECT {columns}
                FROM simulation_history 
                ORDER BY timestamp DESC 
                LIMIT {limit}
             """).fetchall()
             
             history = []
             for r in rows:
                 history.append({
                     "id": r[0],
                     "timestamp": r[1],
                     "model_id": r[2],
                     "ticker": r[3],
                     "return_pct": r[4],
                     "trades_count": r[5],
                     "hit_rate_pct": r[6],
                     "sqn": r[7],
                     "params": json.loads(r[8]) if r[8] else {}
                 })
             return history
    except Exception as e:
        log.error(f"Failed to get history: {e}")
        return []

def get_top_strategies(limit=15, offset=0):
    """Retrieves top strategies sorted by SQN with pagination."""
    try:
        ensure_sim_history_table()
        with duckdb.connect(str(METADATA_DB_PATH), read_only=True) as conn:
             tables = conn.execute("SHOW TABLES").fetchall()
             if not any(t[0] == 'simulation_history' for t in tables):
                 return {"items": [], "total": 0}
             
             # Get total count for pagination
             total_result = conn.execute(
                 "SELECT COUNT(*) FROM simulation_history WHERE trades_count > 5"
             ).fetchone()
             total = total_result[0] if total_result else 0
             
             # Fetch paginated results
             rows = conn.execute("""
                SELECT id, timestamp, model_id, ticker, return_pct, trades_count, hit_rate, sqn, params 
                FROM simulation_history 
                WHERE trades_count > 5 
                ORDER BY sqn DESC 
                LIMIT ? OFFSET ?
             """, [limit, offset]).fetchall()
             
             history = []
             for r in rows:
                 params = {}
                 if r[8]:
                     try:
                         params = json.loads(r[8])
                     except:
                         pass
                 history.append({
                     "id": r[0],
                     "timestamp": r[1],
                     "model_id": r[2],
                     "ticker": r[3],
                     "return_pct": r[4],
                     "trades_count": r[5],
                     "hit_rate_pct": r[6],
                     "sqn": r[7],
                     "params": params
                 })
             return {"items": history, "total": total}
    except Exception as e:
        log.error(f"Failed to get top strategies: {e}")
        return {"items": [], "total": 0}

def delete_all_simulation_history():
    """Deletes all records from simulation_history."""
    try:
        ensure_sim_history_table()
        with duckdb.connect(str(METADATA_DB_PATH)) as conn:
            conn.execute("DELETE FROM simulation_history")
            log.info("Deleted all simulation history.")
        return True
    except Exception as e:
        log.error(f"Failed to delete history: {e}")
        return False

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
            with duckdb.connect(str(METADATA_DB_PATH), read_only=True) as conn:
                tables = conn.execute("SHOW TABLES").fetchall()
                if any(t[0] == 'models' for t in tables):
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
            # Format: Symbol(TF) - Algo - DateTime
            # e.g. "AAPL (1h) | RandomForest | 2023-10-27 14:30"
            created_str = str(meta.get("created_at", ""))
            # Include time if available (first 16 chars = "YYYY-MM-DD HH:MM" or first 19 = "YYYY-MM-DDTHH:MM:SS")
            if len(created_str) >= 16:
                # Handle ISO format with T separator
                date_time_str = created_str[:19].replace('T', ' ')
            elif len(created_str) >= 10:
                date_time_str = created_str[:10]
            else:
                date_time_str = created_str
                
            tf = meta.get("timeframe", "1m")
            sym = meta.get("symbol", "?")
            algo = meta.get("algorithm", "Unknown")
            display = f"{sym} ({tf}) | {algo} | {date_time_str}"
        else:
            display = f"Unknown Model ({mid[:8]})"
            
        result.append({
            "id": mid,
            "name": display,
            "path": info["path"],
            "metadata": meta
        })
        
    # Sort by creation date if available (or name)
    result.sort(key=lambda x: x.get("metadata", {}).get("created_at", "") or "", reverse=True)
    
    return result

def get_available_tickers():
    """Lists available tickers based on feature directories with valid parquet files."""
    log.info(f"Checking for tickers in: {FEATURES_PATH}")
    
    if not FEATURES_PATH.exists():
        log.warning(f"Features path does not exist: {FEATURES_PATH}")
        return []
    
    tickers = []
    for p in FEATURES_PATH.iterdir():
        if p.is_dir():
            # Check if directory has parquet files
            parquet_files = list(p.rglob("*.parquet"))
            if parquet_files:
                tickers.append(p.name)
                log.debug(f"Found ticker {p.name} with {len(parquet_files)} parquet files")
            else:
                log.warning(f"Ticker dir {p.name} exists but has NO parquet files - skipping")
    
    log.info(f"Available tickers with data: {sorted(tickers)}")
    return sorted(tickers)

def load_simulation_data(symbol_str: str, timeframe: str = "1m", options_filter: str = None) -> pd.DataFrame:
    """
    Loads feature data for simulation.
    """
    symbols = [s.strip() for s in symbol_str.split(",")]
    primary_symbol = symbols[0]
    context_symbols = symbols[1:]
    
    log.info(f"Loading simulation data for: {symbol_str}")
    log.info(f"  Primary: {primary_symbol}, Context: {context_symbols}")
    log.info(f"  Features path: {FEATURES_PATH}")
    
    def _load_single(sym):
        path = FEATURES_PATH / sym
        
        log.info(f"Attempting to load {sym} from {path}")
        
        if not path.exists():
            log.error(f"Path does not exist: {path}")
            if FEATURES_PATH.exists():
                existing = [p.name for p in FEATURES_PATH.iterdir() if p.is_dir()]
                log.info(f"Available directories: {existing}")
            return pd.DataFrame()
        
        # Find all parquet files recursively
        files = sorted([str(p) for p in path.rglob("*.parquet")])
        
        if not files:
            log.error(f"No parquet files in {path}")
            # List what IS in the directory
            all_files = list(path.rglob("*"))
            log.info(f"Contents of {path}: {[str(f) for f in all_files[:10]]}")
            return pd.DataFrame()
        
        log.info(f"Found {len(files)} parquet files for {sym}")
        log.debug(f"First 3 files: {files[:3]}")
        
        try:
            # Read parquet files
            df = duckdb.query(f"SELECT * FROM read_parquet({files}) ORDER BY ts").to_df()
            log.info(f"Loaded {len(df)} rows for {sym}, columns: {list(df.columns)[:10]}")
            
            if df.empty:
                log.error(f"Parquet files loaded but DataFrame is empty for {sym}")
            
            return df
            
        except Exception as e:
            log.error(f"DuckDB loading error for {sym}: {e}", exc_info=True)
            raise ValueError(f"Failed to load {sym}: {e}")

    # 1. Load Primary
    df = _load_single(primary_symbol)
    if df.empty:
        available = get_available_tickers()
        
        # Extra diagnostics
        path = FEATURES_PATH / primary_symbol
        if path.exists():
            files = list(path.rglob("*.parquet"))
            log.error(f"Path {path} exists with {len(files)} parquet files but loaded empty DataFrame")
        
        raise ValueError(f"No data for primary symbol {primary_symbol}. Available tickers: {available}")

    # 2. Load and Merge Context
    for ctx_sym in context_symbols:
        ctx_df = _load_single(ctx_sym)
        if ctx_df.empty:
            log.warning(f"Context symbol {ctx_sym} has no data, skipping")
            continue
            
        cols_to_rename = {c: f"{c}_{ctx_sym}" for c in ctx_df.columns if c != "ts"}
        ctx_df = ctx_df.rename(columns=cols_to_rename)
        
        df = pd.merge(df, ctx_df, on="ts", how="inner")
        log.info(f"Merged {ctx_sym}, now {len(df)} rows")
        
    # 3. Resample if needed
    if timeframe and timeframe != "1m":
        log.info(f"Resampling to {timeframe}")
        df = df.set_index("ts").sort_index()
        agg_dict = {}
        for col in df.columns:
            if "split" in col or "data_split" in col:
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
        df = df.dropna(subset=[c for c in df.columns if "close" in c][:1])
        df = df.reset_index()
        log.info(f"After resampling: {len(df)} rows")

    return df

def _prepare_simulation_inputs(model_id: str, ticker: str, 
                             min_prediction_threshold: float = 0.0, 
                             enable_z_score_check: bool = False, 
                             volatility_normalization: bool = False):
    """
    Shared logic to load Model, Data, and prepare Features/Predictions/Signals.
    Returns (df_sim, X, model)
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
             # Try to select target_transform first
             try:
                 row = conn.execute("SELECT symbol, timeframe, data_options, target_transform FROM models WHERE id = ?", [model_id.replace(".joblib", "")]).fetchone()
                 if row:
                     metadata = {"symbol": row[0], "timeframe": row[1], "data_options": row[2], "target_transform": row[3]}
             except Exception:
                 # Fallback for old schema
                 row = conn.execute("SELECT symbol, timeframe, data_options FROM models WHERE id = ?", [model_id.replace(".joblib", "")]).fetchone()
                 if row:
                     metadata = {"symbol": row[0], "timeframe": row[1], "data_options": row[2]}

    trained_symbol_str = metadata.get("symbol", ticker) 
    timeframe = metadata.get("timeframe", "1m")
    data_options = metadata.get("data_options")
    target_transform = metadata.get("target_transform", "none")
    
    trained_context = trained_symbol_str.split(",")[1:]
    
    target_load_str = ticker 
    if trained_context:
        target_load_str = ",".join([ticker] + [s.strip() for s in trained_context])
        
    log.info(f"Loading model form {model_path}")
    model = joblib.load(model_path)
    
    log.info(f"Loading data for {target_load_str} (TF={timeframe}, Opts={data_options})")
    df = load_simulation_data(target_load_str, timeframe, data_options)
    
    if df.empty:
        raise ValueError(f"No data for {target_load_str}")

    # Prepare features
    drop_cols = ["target", "ts", "symbol", "date", "source", "options", "target_col_shifted", "dt"]
    for col in df.columns:
        if "train" in col.lower() or "test" in col.lower():
            if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
                 drop_cols.append(col)

    df_numeric = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in df_numeric.columns if c not in drop_cols]
    
    if not feature_cols:
        raise ValueError("No feature columns found")

    X = df[feature_cols].copy()
    
    if hasattr(model, "feature_names_in_"):
        required_features = list(model.feature_names_in_)
        missing = [c for c in required_features if c not in X.columns]
        if missing:
             for m in missing:
                 if m in df.columns:
                     X[m] = df[m]
        
        missing = [c for c in required_features if c not in X.columns]
        if missing:
            raise ValueError(f"Model requires features not present in data: {missing}")
        
        X = X[required_features]
    
    X = X.fillna(0)
    
    # -------------------------------------------------------------
    # 1. Z-Score Scaling Checks (Outlier Removal)
    # -------------------------------------------------------------
    if enable_z_score_check:
        # Calculate Z-scores (skip constant columns to avoid NaN)
        # Simple Z-score: (x - mean) / std
        # Handle division by zero for constant cols
        desc = X.describe().T
        std = desc['std']
        mean = desc['mean']
        
        # Only check columns with std > 0
        valid_cols = std[std > 1e-9].index
        
        if len(valid_cols) > 0:
            X_valid = X[valid_cols]
            z_scores = (X_valid - mean[valid_cols]) / std[valid_cols]
            
            # Identify outliers (> 4 sigma)
            outliers = (z_scores.abs() > 4).any(axis=1)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                log.info(f"Z-Score Check: Dropping {n_outliers} outlier rows (Sigma > 4).")
                X = X[~outliers]
                df = df[~outliers]

    # -------------------------------------------------------------
    # 2. Volatility Normalization (Rescaling)
    # -------------------------------------------------------------
    if volatility_normalization:
        log.info("Applying Volatility Normalization (StandardScaler) to Simulation Data.")
        scaler = StandardScaler()
        # Keep DataFrame structure
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Predict
    try:
        preds = model.predict(X)
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        raise e

    df_sim = df.copy()
    df_sim["prediction"] = preds
    
    is_classifier = hasattr(model, "predict_proba") or len(np.unique(preds)) <= 2

    if is_classifier:
        df_sim["signal"] = df_sim["prediction"].apply(lambda x: 1 if x == 1 else 0)
        df_sim["pred_dir"] = df_sim["signal"]
    else:
        # Regression Logic with Threshold
        # Check target_transform to decide comparison logic
        # If model predicts Returns (Log or Pct), compare directly against threshold (> 0 means Up)
        # If model predicts Price (None), compare (Price - Close)/Close against threshold
        
        # Note: target_transform variable is available in this scope? No, need to pass it or extract.
        # But wait, this code is inside the function that returns it.
        # Let's use the local variable logic.
        
        is_return_pred = target_transform in ["log_return", "pct_change"]
        
        if is_return_pred:
            # Prediction IS the expected return.
            # Buy if Pred > Threshold
            # Sell if Pred < -Threshold (or just <= 0? User asked for threshold).
            # Usually threshold implies "Magnitude of confidence".
            # If Threshold = 0.0005, then Buy if > 0.0005. Sell if < 0. (Or < -0.0005?)
            # Standard logic: > 0 for Buy. Threshold logic: > Threshold.
            
            df_sim["signal"] = df_sim["prediction"].apply(lambda x: 1 if x > min_prediction_threshold else 0)
            
            # Direction for Hit Rate: > 0 is UP.
            df_sim["pred_dir"] = (df_sim["prediction"] > 0).astype(int)
            
        else:
            # Prediction IS Price.
            # (Pred - Close) / Close > Threshold
            df_sim["signal"] = df_sim.apply(
                lambda row: 1 if ((row["prediction"] - row["close"]) / row["close"]) > min_prediction_threshold else 0, 
                axis=1
            )
            df_sim["pred_dir"] = (df_sim["prediction"] > df_sim["close"]).astype(int)

    df_sim["next_close"] = df_sim["close"].shift(-1)
    df_sim["actual_dir"] = (df_sim["next_close"] > df_sim["close"]).astype(int)
    
    # Hit: 1 if Correct, 0 if Incorrect
    df_sim["hit"] = (df_sim["pred_dir"] == df_sim["actual_dir"]).astype(int)
    df_sim.loc[df_sim.index[-1], "hit"] = 0 
    
    df_sim["rolling_hit_rate"] = df_sim["hit"].rolling(window=20).mean() # For display
    
    return df_sim, X, model, target_transform


def train_trading_bot(model_id: str, ticker: str, 
                      min_prediction_threshold: float = 0.0,
                      enable_z_score_check: bool = False,
                      volatility_normalization: bool = False):
    """
    Trains a secondary model (Bot) to predict if the Base Model's signal will hit.
    """
    try:
        log.info(f"Training Bot for Model={model_id}, Ticker={ticker}")
        df_sim, X, _, _ = _prepare_simulation_inputs(
            model_id, ticker,
            min_prediction_threshold=min_prediction_threshold,
            enable_z_score_check=enable_z_score_check,
            volatility_normalization=volatility_normalization
        )
        
        # Target: Is the base model correct?
        # Labels: 'hit' (1=Correct, 0=Incorrect)
        # We drop the last row because target is future dependent
        X_train = X.iloc[:-1]
        y_train = df_sim["hit"].iloc[:-1]
        
        if len(X_train) < 50:
            return {"status": "error", "message": "Not enough data to train bot"}
            
        # Train Random Forest
        # We use a relatively simple model to avoid overfitting too much on noise
        bot = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        bot.fit(X_train, y_train)
        
        # Save
        if not BOTS_DIR.exists():
            BOTS_DIR.mkdir(parents=True, exist_ok=True)
            
        bot_path = BOTS_DIR / f"{model_id}.joblib"
        joblib.dump(bot, bot_path)
        
        score = bot.score(X_train, y_train)
        log.info(f"Bot trained. Accuracy: {score:.2f}")
        
        return {"status": "success", "accuracy": score, "path": str(bot_path)}
        
    except Exception as e:
        log.error(f"Bot training failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def run_simulation(model_id: str, ticker: str, initial_cash: float, use_bot: bool = False,
                   min_prediction_threshold: float = 0.0,
                   enable_z_score_check: bool = False,
                   volatility_normalization: bool = False,
                   regime_col: str = None,
                   allowed_regimes: list = None,
                   save_to_history: bool = True,
                   enable_slippage: bool = True,
                   slippage_bars: int = 4,
                   transaction_fee: float = 0.02):
    """
    Runs a backtest simulation with realistic trading costs.
    
    Args:
        enable_slippage: If True, trades execute after slippage_bars delay
        slippage_bars: Number of bars to wait before execution (default 4)
        transaction_fee: Flat fee per trade in dollars (default $0.02)
    """
    log.info("="*60)
    log.info(f"Starting Simulation: {ticker} | Model: {model_id[:12]}...")
    log.info("="*60)
    
    # 1. Prepare Data
    log.info(f"Initial Capital: ${initial_cash:,.2f}")
    log.info(f"Prediction Threshold: {min_prediction_threshold:.4f} (signals require >{min_prediction_threshold*100:.2f}% confidence)")
    
    df_sim, X, base_model, target_transform = _prepare_simulation_inputs(
        model_id, ticker,
        min_prediction_threshold=min_prediction_threshold,
        enable_z_score_check=enable_z_score_check,
        volatility_normalization=volatility_normalization
    )
    
    log.info(f"Loaded {len(df_sim)} bars for backtesting")
    log.info(f"Date Range: {df_sim['ts'].min()} to {df_sim['ts'].max()}")
    
    # 2. Apply Bot Logic (if enabled)
    if use_bot:
        bot_path = BOTS_DIR / f"{model_id}.joblib"
        if bot_path.exists():
            log.info("Applying Trading Bot Filter (ML-based signal confirmation)...")
            bot = joblib.load(bot_path)
            
            try:
                bot_probs = bot.predict_proba(X)[:, 1]
                
                new_signals = []
                orig_signals = df_sim["signal"].values
                filtered_count = 0
                for i, prob in enumerate(bot_probs):
                    base_sig = orig_signals[i]
                    if prob > 0.55:
                        new_signals.append(base_sig)
                    else:
                        new_signals.append(-1)
                        if base_sig == 1:
                            filtered_count += 1
                        
                df_sim["signal"] = new_signals
                log.info(f"Trading Bot filtered out {filtered_count} low-confidence signals")
                
            except Exception as e:
                log.warning(f"Bot prediction failed, falling back to base model: {e}")
        else:
             log.warning("Bot enabled but no bot model found. Run 'Train Bot' first.")
    else:
        log.info("Trading Bot: DISABLED (using raw model predictions)")

    # 3. Regime Filter Logic
    if regime_col and allowed_regimes is not None:
        if regime_col in df_sim.columns:
            log.info(f"Applying Regime Filter: {regime_col} MUST BE in {allowed_regimes}")
            
            def apply_regime_gate(row):
                current_regime = row[regime_col]
                if current_regime in allowed_regimes:
                    return row["signal"]
                else:
                    return 0

            pre_filter_signals = df_sim["signal"].sum()
            df_sim["signal"] = df_sim.apply(apply_regime_gate, axis=1)
            post_filter_signals = df_sim["signal"].sum()
            blocked_signals = pre_filter_signals - post_filter_signals
            
            log.info(f"Regime Filter blocked {blocked_signals} signals in unfavorable market conditions")
        else:
            log.warning(f"Regime column {regime_col} not found in data. Ignoring filter.")
    else:
        log.info("Regime Filter: DISABLED (trading in all market conditions)")

    # 4. Slippage Simulation (Delayed Execution)
    if enable_slippage and slippage_bars > 0:
        log.info(f"SLIPPAGE MODEL: {slippage_bars}-bar execution delay with midpoint pricing")
        log.info(f"  -> Orders fill at mean(open, close) of bar T+{slippage_bars}")
        log.info(f"  -> Simulates market impact, order routing delays, and partial fills")
        
        # Create delayed execution price column
        # Price = mean of (open + close) at execution bar
        df_sim["exec_price"] = ((df_sim["open"] + df_sim["close"]) / 2).shift(-slippage_bars)
        
        # Shift signals forward to align with execution
        df_sim["exec_signal"] = df_sim["signal"].copy()
        df_sim["signal"] = df_sim["signal"].shift(slippage_bars).fillna(0).astype(int)
        
        # Use exec_price for trading, fallback to close if NaN (end of data)
        df_sim["trade_price"] = df_sim["exec_price"].fillna(df_sim["close"])
    else:
        log.info("SLIPPAGE MODEL: DISABLED (instant fills at close price - UNREALISTIC)")
        # No slippage: use close price immediately
        df_sim["trade_price"] = df_sim["close"]

    # 5. Transaction Cost Model
    log.info(f"TRANSACTION COSTS: ${transaction_fee:.2f} per trade (entry + exit = ${transaction_fee*2:.2f} round-trip)")
    log.info(f"  -> Includes: SEC fees, exchange fees, clearing fees, and rounding")

    # Walk forward simulation with costs
    log.info("-"*60)
    log.info("Beginning Walk-Forward Backtest...")
    log.info("-"*60)
    
    cash = initial_cash
    shares = 0
    portfolio_values = []
    trades = []
    last_buy_price = 0.0
    total_fees = 0.0
    
    # Initialize metrics to defaults to avoid UnboundLocalError if no trades occur
    winning_trades = []
    losing_trades = []
    win_rate = 0.0

    if len(trades) > 0:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(winning_trades) / len(trades)

    log.info(f"Total Round-Trip Trades: {len(trades)}")
    log.info(f"Win Rate: {win_rate*100:.1f}% ({len(winning_trades)} wins, {len(losing_trades)} losses)")
    
    # Benchmark: Buy and Hold
    initial_price = df_sim.iloc[0]["close"]
    benchmark_shares = initial_cash / initial_price
    
    log.info(f"Benchmark Strategy: Buy & Hold {benchmark_shares:.2f} shares at ${initial_price:.2f}")
    
    for idx, row in df_sim.iterrows():
        price = row["trade_price"]  # Use slippage-adjusted price
        ts = row["ts"]
        signal = row["signal"]
        
        action = None
        
        if signal == 1 and shares == 0:
            # BUY: Whole shares only
            # Account for transaction fee in available cash
            possible_shares = int((cash - transaction_fee) // price)
            if possible_shares > 0:
                shares = possible_shares
                cost = shares * price
                cash -= (cost + transaction_fee)
                total_fees += transaction_fee
                last_buy_price = price
                action = "BUY"
                log.debug(f"{ts}: BUY {shares} shares @ ${price:.2f} (fee: ${transaction_fee:.2f}, cash remaining: ${cash:.2f})")
                trades.append({
                    "ts": ts,
                    "type": "BUY",
                    "price": price,
                    "shares": shares,
                    "value": cost,
                    "fee": transaction_fee,
                    "pnl": 0.0,
                    "pnl_pct": 0.0
                })
        elif signal == 0 and shares > 0:
            # SELL
            proceeds = shares * price
            
            # Calculate PnL (before fees)
            pnl = proceeds - (shares * last_buy_price)
            pnl_pct = (price - last_buy_price) / last_buy_price * 100
            
            # Net PnL after both entry and exit fees
            net_pnl = pnl - (2 * transaction_fee) # Entry + Exit fees
            
            # Categorize trade
            if net_pnl > 0:
                winning_trades.append(net_pnl)
            else:
                losing_trades.append(abs(net_pnl))
            
            # Deduct exit fee
            cash += (proceeds - transaction_fee)
            total_fees += transaction_fee
            
            log.debug(f"{ts}: SELL {shares} shares @ ${price:.2f} | Net P&L: ${net_pnl:+.2f} ({pnl_pct:+.2f}%) | Equity: ${cash:.2f}")
            
            shares = 0
            action = "SELL"
            trades.append({
                "ts": ts,
                "type": "SELL",
                "price": price,
                "shares": 0,
                "value": proceeds,
                "fee": transaction_fee,
                "pnl": net_pnl,
                "pnl_pct": (net_pnl / (shares * last_buy_price)) * 100 if shares > 0 else 0
            })
            
        current_val = cash + (shares * price)
        portfolio_values.append(current_val)
    
    df_sim["strategy_equity"] = portfolio_values
    df_sim["benchmark_equity"] = df_sim["close"] * benchmark_shares
    
    # Calculate Quant Metrics
    total_round_trips = len(winning_trades) + len(losing_trades)
    
    log.info("-"*60)
    log.info("Backtest Complete - Calculating Performance Metrics...")
    log.info("-"*60)
    
    # Trade Expectancy
    if total_round_trips > 0:
        win_rate = len(winning_trades) / total_round_trips
        loss_rate = len(losing_trades) / total_round_trips
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    else:
        expectancy = 0.0
        avg_win = 0.0
        avg_loss = 0.0
    
    # Profit Factor
    gross_profit = sum(winning_trades) if winning_trades else 0.0
    gross_loss = sum(losing_trades) if losing_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    
    # System Quality Number (SQN)
    if total_round_trips > 1:
        all_pnl = [t["pnl"] for t in trades if t["type"] == "SELL"]
        if len(all_pnl) > 1:
            avg_pnl = np.mean(all_pnl)
            std_pnl = np.std(all_pnl, ddof=1)
            sqn = (avg_pnl / std_pnl) * np.sqrt(len(all_pnl)) if std_pnl > 0 else 0.0
        else:
            sqn = 0.0
    else:
        sqn = 0.0
    
    # Results
    valid_hits = df_sim["hit"].iloc[:-1]
    hit_rate_pct = (valid_hits.sum() / len(valid_hits) * 100) if len(valid_hits) > 0 else 0.0

    final_equity = df_sim["strategy_equity"].iloc[-1]
    strategy_return = (final_equity - initial_cash) / initial_cash * 100
    benchmark_return = (df_sim["benchmark_equity"].iloc[-1] - initial_cash) / initial_cash * 100
    
    log.info(f"Total Round-Trip Trades: {total_round_trips}")
    log.info(f"Win Rate: {win_rate*100:.1f}% ({len(winning_trades)} wins, {len(losing_trades)} losses)")
    log.info(f"Average Win: ${avg_win:.2f} | Average Loss: ${avg_loss:.2f}")
    log.info(f"Expectancy (Avg P&L per trade): ${expectancy:.2f}")
    log.info(f"Profit Factor (Gross Profit / Gross Loss): {profit_factor:.2f}")
    log.info(f"System Quality Number (SQN): {sqn:.2f} {'(Excellent)' if sqn > 3 else '(Good)' if sqn > 2 else '(Fair)' if sqn > 1 else '(Poor)'}")
    log.info(f"Total Fees Paid: ${total_fees:.2f} ({(total_fees/initial_cash)*100:.2f}% of capital)")
    log.info(f"Hit Rate (Directional Accuracy): {hit_rate_pct:.1f}%")
    log.info("-"*60)
    log.info(f"FINAL RESULTS:")
    log.info(f"  Strategy Return: {strategy_return:+.2f}% (${final_equity:,.2f})")
    log.info(f"  Benchmark Return: {benchmark_return:+.2f}% (${df_sim['benchmark_equity'].iloc[-1]:,.2f})")
    log.info(f"  Alpha (Outperformance): {strategy_return - benchmark_return:+.2f}%")
    log.info("="*60)

    stats = {
        "start_date": df_sim["ts"].min().isoformat(),
        "end_date": df_sim["ts"].max().isoformat(),
        "final_strategy_value": final_equity,
        "final_benchmark_value": df_sim["benchmark_equity"].iloc[-1],
        "strategy_return_pct": strategy_return,
        "benchmark_return_pct": benchmark_return,
        "total_trades": len(trades),
        "total_fees": total_fees,
        "avg_fee_per_trade": total_fees / len(trades) if len(trades) > 0 else 0,
        "hit_rate_pct": hit_rate_pct,
        "bot_active": use_bot,
        "slippage_enabled": enable_slippage,
        "slippage_bars": slippage_bars if enable_slippage else 0,
        # NEW: Quant Metrics
        "expectancy": expectancy,
        "sqn": sqn,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_rate": (len(winning_trades) / total_round_trips * 100) if total_round_trips > 0 else 0.0,
        "profit_factor": profit_factor
    }

    # Chart Data
    chart_data = df_sim[["ts", "strategy_equity", "benchmark_equity", "rolling_hit_rate"]].fillna(0).copy()
    chart_data["ts"] = chart_data["ts"].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save History
    run_params = {
        "initial_cash": initial_cash,
        "use_bot": use_bot,
        "min_prediction_threshold": min_prediction_threshold,
        "enable_z_score_check": enable_z_score_check,
        "volatility_normalization": volatility_normalization,
        "slippage_bars": slippage_bars if enable_slippage else 0,
        "transaction_fee": transaction_fee
    }
    
    if save_to_history:
        save_simulation_history(model_id, ticker, stats, run_params)

    return {
        "stats": stats,
        "trades": trades,
        "chart_data": chart_data.to_dict(orient="records")
    }

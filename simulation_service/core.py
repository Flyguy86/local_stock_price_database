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

log = logging.getLogger("simulation.core")

# Settings / Config from environment or defaults
MODELS_DIR = Path("/app/data/models")
BOTS_DIR = Path("/app/data/models/bots")
FEATURES_PATH = Path("/app/data/features_parquet")
METADATA_DB_PATH = Path("/app/data/duckdb/models.db")

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
                    params JSON
                )
            """)
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
                (id, timestamp, model_id, ticker, return_pct, trades_count, hit_rate, params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record_id, 
                ts, 
                model_id, 
                ticker, 
                stats.get('strategy_return_pct', 0.0),
                stats.get('total_trades', 0),
                stats.get('hit_rate_pct', 0.0),
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
                 
             rows = conn.execute(f"""
                SELECT id, timestamp, model_id, ticker, return_pct, trades_count, hit_rate, params 
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
                     "hit_rate": r[6],
                     "params": json.loads(r[7]) if r[7] else {}
                 })
             return history
    except Exception as e:
        log.error(f"Failed to get history: {e}")
        return []

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
             row = conn.execute("SELECT symbol, timeframe, data_options FROM models WHERE id = ?", [model_id.replace(".joblib", "")]).fetchone()
             if row:
                 metadata = {"symbol": row[0], "timeframe": row[1], "data_options": row[2]}

    trained_symbol_str = metadata.get("symbol", ticker) 
    timeframe = metadata.get("timeframe", "1m")
    data_options = metadata.get("data_options")
    
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
        # (Predicted Price - Close) / Close > Threshold
        df_sim["signal"] = df_sim.apply(
            lambda row: 1 if (row["prediction"] - row["close"]) / row["close"] > min_prediction_threshold else 0, 
            axis=1
        )
        df_sim["pred_dir"] = df_sim["signal"]

    df_sim["next_close"] = df_sim["close"].shift(-1)
    df_sim["actual_dir"] = (df_sim["next_close"] > df_sim["close"]).astype(int)
    
    # Hit: 1 if Correct, 0 if Incorrect
    df_sim["hit"] = (df_sim["pred_dir"] == df_sim["actual_dir"]).astype(int)
    df_sim.loc[df_sim.index[-1], "hit"] = 0 
    
    df_sim["rolling_hit_rate"] = df_sim["hit"].rolling(window=20).mean() # For display
    
    return df_sim, X, model


def train_trading_bot(model_id: str, ticker: str, 
                      min_prediction_threshold: float = 0.0,
                      enable_z_score_check: bool = False,
                      volatility_normalization: bool = False):
    """
    Trains a secondary model (Bot) to predict if the Base Model's signal will hit.
    """
    try:
        log.info(f"Training Bot for Model={model_id}, Ticker={ticker}")
        df_sim, X, _ = _prepare_simulation_inputs(
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
                   volatility_normalization: bool = False):
    """
    Runs a backtest simulation.
    """
    # 1. Prepare Data
    df_sim, X, base_model = _prepare_simulation_inputs(
        model_id, ticker,
        min_prediction_threshold=min_prediction_threshold,
        enable_z_score_check=enable_z_score_check,
        volatility_normalization=volatility_normalization
    )
    
    # 2. Apply Bot Logic (if enabled)
    if use_bot:
        bot_path = BOTS_DIR / f"{model_id}.joblib"
        if bot_path.exists():
            log.info("Applying Trading Bot...")
            bot = joblib.load(bot_path)
            
            # Bot predicts Probability of "Hit" (Class 1)
            # Input X is same as base model
            try:
                bot_probs = bot.predict_proba(X)[:, 1]
                
                # Logic: Only trade if Bot is > 50% confident that Base Model is Correct
                # Base Signal: 1 (Buy) or 0 (Sell)
                # New Signal:
                # If Base=Buy AND Bot=Correct -> Buy
                # If Base=Sell AND Bot=Correct -> Sell
                # If Bot=Incorrect -> Hold (Signal usually 0 means Sell... we need a distinct HOLD signal?)
                # 
                # Our simulation loop treats Signal 0 as Sell/Clear Position.
                # It treats Signal 1 as Buy.
                # If we want to HOLD, we need to pass a "None" signal?
                # The current loop:
                # if signal == 1: Buy
                # elif signal == 0: Sell
                # else: Hold (implied if logic doesn't match)
                
                # So we can introduce -1 or 2 as Hold?
                # Actually, if signal is not 1 or 0, it does nothing.
                
                # Let's map:
                # Buy Confirmed -> 1
                # Sell Confirmed -> 0
                # Uncertain -> -1 (Hold)
                
                new_signals = []
                orig_signals = df_sim["signal"].values
                for i, prob in enumerate(bot_probs):
                    base_sig = orig_signals[i]
                    if prob > 0.55: # Threshold > 50%
                        # Bot agrees model is correct
                        new_signals.append(base_sig)
                    else:
                        # Bot thinks model is WRONG.
                        # If Model said Buy (and is wrong), Price goes Down -> We should Sell? 
                        # Or just Stay Out?
                        # Usually "Trading Bot" means "Filter out bad trades". So Stay Out / Hold.
                        new_signals.append(-1) 
                        
                df_sim["signal"] = new_signals
                
            except Exception as e:
                log.warning(f"Bot prediction failed, falling back to base model: {e}")
        else:
             log.warning("Bot enabled but no bot model found. Run 'Train Bot' first.")

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
        
        action = None
        # Signal 1 = Buy
        # Signal 0 = Sell
        # Signal -1 = Hold (New)
        
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
        # If signal is -1 (Hold), do nothing.
            
        current_val = cash + (shares * price)
        portfolio_values.append(current_val)
    
    df_sim["strategy_equity"] = portfolio_values
    df_sim["benchmark_equity"] = df_sim["close"] * benchmark_shares
    
    # Results
    # Calc Hit Rate (exclude last row which has no next_close)
    valid_hits = df_sim["hit"].iloc[:-1]
    hit_rate_pct = (valid_hits.sum() / len(valid_hits) * 100) if len(valid_hits) > 0 else 0.0

    stats = {
        "start_date": df_sim["ts"].min().isoformat(),
        "end_date": df_sim["ts"].max().isoformat(),
        "final_strategy_value": df_sim["strategy_equity"].iloc[-1],
        "final_benchmark_value": df_sim["benchmark_equity"].iloc[-1],
        "strategy_return_pct": (df_sim["strategy_equity"].iloc[-1] - initial_cash) / initial_cash * 100,
        "benchmark_return_pct": (df_sim["benchmark_equity"].iloc[-1] - initial_cash) / initial_cash * 100,
        "total_trades": len(trades),
        "hit_rate_pct": hit_rate_pct,
        "bot_active": use_bot
    }

    # Chart Data (Downsample if needed?)
    chart_data = df_sim[["ts", "strategy_equity", "benchmark_equity", "rolling_hit_rate"]].fillna(0).copy()
    chart_data["ts"] = chart_data["ts"].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save History
    run_params = {
        "initial_cash": initial_cash,
        "use_bot": use_bot,
        "min_prediction_threshold": min_prediction_threshold,
        "enable_z_score_check": enable_z_score_check,
        "volatility_normalization": volatility_normalization
    }
    save_simulation_history(model_id, ticker, stats, run_params)

    return {
        "stats": stats,
        "trades": trades,
        "chart_data": chart_data.to_dict(orient="records")
    }

from __future__ import annotations
import json
import logging
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import tempfile
import shutil
import hashlib

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger("feature_service")


def _has_table(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = current_schema() AND lower(table_name) = lower(?)
            LIMIT 1
            """,
            [name],
        ).fetchone()
        return bool(row)
    except duckdb.Error:
        return False


def list_symbols(src_conn: duckdb.DuckDBPyConnection) -> list[str]:
    if not _has_table(src_conn, "bars"):
        return []
    rows = src_conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol").fetchall()
    return [row[0] for row in rows]


def fetch_bars(src_conn: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    if not _has_table(src_conn, "bars"):
        return pd.DataFrame()
    return src_conn.execute(
        """
        SELECT ts, open, high, low, close, volume, vwap, trade_count
        FROM bars
        WHERE symbol = ?
        ORDER BY ts
        """,
        [symbol],
    ).fetch_df()


def fetch_earnings(src_conn: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    if not _has_table(src_conn, "earnings"):
        return pd.DataFrame()
    return src_conn.execute(
        """
        SELECT announce_date
        FROM earnings
        WHERE symbol = ?
        ORDER BY announce_date
        """,
        [symbol],
    ).fetch_df()


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.sort_values("ts")
    out = out.drop_duplicates(subset=["ts"], keep="last")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[out["volume"].fillna(0) >= 0]

    # Normalize dtypes to avoid NAType -> float conversion errors when inserting/writing parquet
    out["open"] = out["open"].astype("float64")
    out["high"] = out["high"].astype("float64")
    out["low"] = out["low"].astype("float64")
    out["close"] = out["close"].astype("float64")
    out["volume"] = out["volume"].fillna(0).astype("int64")
    if "trade_count" in out.columns:
        out["trade_count"] = out["trade_count"].fillna(0).astype("int64")
    if "vwap" in out.columns:
        out["vwap"] = out["vwap"].ffill().fillna(0.0).astype("float64")
    return out.reset_index(drop=True)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def _calculate_vix_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates specific market regime metrics on VIXY data."""
    if df.empty:
        return pd.DataFrame()
        
    out = df.copy()
    out = out.sort_values("ts")
    
    # 1. Log Returns
    out["vix_log_ret"] = np.log(out["close"] / out["close"].shift(1)).fillna(0.0)
    
    # 2. Z-Score Spike (Window of 20 minutes)
    roll_mean = out["vix_log_ret"].rolling(20, min_periods=1).mean()
    roll_std = out["vix_log_ret"].rolling(20, min_periods=1).std(ddof=0)
    out["vix_z_score"] = (out["vix_log_ret"] - roll_mean) / roll_std.replace(0, np.nan)
    out["vix_z_score"] = out["vix_z_score"].fillna(0.0)

    # 3. Volume Spike (Relative to 1-hour average)
    out["vix_rel_vol"] = out["volume"] / out["volume"].rolling(60, min_periods=1).mean().replace(0, np.nan)
    out["vix_rel_vol"] = out["vix_rel_vol"].fillna(0.0)

    # 4. Range Expansion (Current candle vs 15-min ATR)
    prev_c = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_c).abs(),
        (out["low"] - prev_c).abs()
    ], axis=1).max(axis=1)
    
    tr_mean = tr.rolling(15, min_periods=1).mean()
    out["vix_atr_ratio"] = tr / tr_mean.replace(0, np.nan)
    out["vix_atr_ratio"] = out["vix_atr_ratio"].fillna(0.0)
    
    return out[["ts", "vix_log_ret", "vix_z_score", "vix_rel_vol", "vix_atr_ratio"]]


def _calculate_features_for_chunk(df: pd.DataFrame, opts: dict) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("ts")

    # Feature toggles
    use_sma = opts.get("use_sma", True)
    use_bb = opts.get("use_bb", True)
    use_rsi = opts.get("use_rsi", True)
    use_macd = opts.get("use_macd", True)
    use_atr = opts.get("use_atr", True)
    use_vol = opts.get("use_vol", True)
    use_time = opts.get("use_time", True)

    # Returns and moving averages
    out["return_1m"] = out["close"].pct_change().fillna(0.0)
    out["lag_1_close"] = out["close"].shift(1)  # New lag feature

    # Distance from VWAP (Mean Reversion)
    # Formula: (Close - VWAP) / VWAP
    out["dist_vwap"] = (out["close"] - out["vwap"]) / out["vwap"].replace(0, np.nan)
    out["dist_vwap"] = out["dist_vwap"].fillna(0.0)

    # Intraday Intensity ("Smart Money" Index)
    # Formula: Intensity = ((2 * Close - High - Low) / (High - Low)) * Volume
    # High volume near Close implies institutional accumulation/dumping
    denom_ii = (out["high"] - out["low"]).replace(0, np.nan)
    term1_ii = (2 * out["close"] - out["high"] - out["low"]) / denom_ii
    out["intraday_intensity"] = term1_ii.fillna(0.0) * out["volume"]

    # Volatility-Adjusted Momentum (Z-Score)
    # Formula: (Current Return - Mean Return 20) / Std Dev 20
    roll_mean_ret = out["return_1m"].rolling(window=20, min_periods=1).mean()
    roll_std_ret = out["return_1m"].rolling(window=20, min_periods=1).std(ddof=0)
    out["vol_adj_mom_20"] = (out["return_1m"] - roll_mean_ret) / roll_std_ret.replace(0, np.nan)
    out["vol_adj_mom_20"] = out["vol_adj_mom_20"].fillna(0.0)

    # 1. Log Returns
    # ln(close / prev_close)
    out["log_return_1m"] = np.log(out["close"] / out["close"].shift(1)).fillna(0.0)

    # 2. Return Z-Score (Volatility Spike) - Window 20
    # Formula: (Log Return - Mean Log Return 20) / Std Log Return 20
    roll_mean_log = out["log_return_1m"].rolling(window=20, min_periods=1).mean()
    roll_std_log = out["log_return_1m"].rolling(window=20, min_periods=1).std(ddof=0)
    out["return_z_score_20"] = (out["log_return_1m"] - roll_mean_log) / roll_std_log.replace(0, np.nan)
    out["return_z_score_20"] = out["return_z_score_20"].fillna(0.0)

    # 3. Relative Volume (Volume Spike) - Window 60
    # Formula: Volume / Avg Volume 60
    vol_mean_60 = out["volume"].rolling(window=60, min_periods=1).mean()
    out["vol_ratio_60"] = out["volume"] / vol_mean_60.replace(0, np.nan)
    out["vol_ratio_60"] = out["vol_ratio_60"].fillna(0.0)

    # 4. ATR Ratio (Range Expansion) - Window 15
    # True Range = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    prev_c = out["close"].shift(1)
    # We can use vector max: max(A, B, C)
    tr_series = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_c).abs(),
        (out["low"] - prev_c).abs()
    ], axis=1).max(axis=1)
    
    tr_mean_15 = tr_series.rolling(window=15, min_periods=1).mean()
    out["atr_ratio_15"] = tr_series / tr_mean_15.replace(0, np.nan)
    out["atr_ratio_15"] = out["atr_ratio_15"].fillna(0.0)
    
    if use_sma:
        out["sma_close_5"] = out["close"].rolling(window=5, min_periods=1).mean()
        out["sma_close_20"] = out["close"].rolling(window=20, min_periods=1).mean()
    else:
        out["sma_close_5"] = np.nan
        out["sma_close_20"] = np.nan

    # Bollinger Bands
    if use_bb:
        rolling_mean_20 = out["close"].rolling(window=20, min_periods=1).mean()
        rolling_std_20 = out["close"].rolling(window=20, min_periods=1).std(ddof=0)
        out["bb_upper_20_2"] = rolling_mean_20 + 2 * rolling_std_20
        out["bb_lower_20_2"] = rolling_mean_20 - 2 * rolling_std_20
    else:
        out["bb_upper_20_2"] = np.nan
        out["bb_lower_20_2"] = np.nan

    # RSI
    if use_rsi:
        out["rsi_14"] = _rsi(out["close"], period=14)
    else:
        out["rsi_14"] = np.nan

    # MACD
    if use_macd:
        ema12 = out["close"].ewm(span=12, adjust=False).mean()
        ema26 = out["close"].ewm(span=26, adjust=False).mean()
        out["macd_line"] = ema12 - ema26
        out["macd_signal"] = out["macd_line"].ewm(span=9, adjust=False).mean()
        out["macd_hist"] = out["macd_line"] - out["macd_signal"]
    else:
        out["macd_line"] = np.nan
        out["macd_signal"] = np.nan
        out["macd_hist"] = np.nan

    # ATR
    if use_atr:
        prev_close = out["close"].shift(1)
        tr = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr_14"] = tr.rolling(window=14, min_periods=1).mean()
    else:
        out["atr_14"] = np.nan

    # Volume features
    if use_vol:
        out["vol_sma_20"] = out["volume"].rolling(window=20, min_periods=1).mean()
        out["volume_change"] = out["volume"].pct_change().fillna(0.0) # New volume feature
    else:
        out["vol_sma_20"] = np.nan
        out["volume_change"] = np.nan

    # Time-based features
    if use_time:
        ts = pd.to_datetime(out["ts"], utc=True)
        out["time_of_day_min"] = (ts.dt.hour * 60 + ts.dt.minute).astype("int64")
        out["day_of_week"] = ts.dt.dayofweek.astype("int64")
        out["day_of_month"] = ts.dt.day.astype("int64")
        out["month"] = ts.dt.month.astype("int64")
    else:
        out["time_of_day_min"] = 0
        out["day_of_week"] = 0
        out["day_of_month"] = 0
        out["month"] = 0

    # Earnings features
    earnings_df = opts.get("earnings_df")
    if earnings_df is not None and not earnings_df.empty:
        # Prepare earnings timestamps (midnight UTC)
        # Handle potential date objects or strings
        earnings_ts = pd.to_datetime(earnings_df["announce_date"])
        if earnings_ts.dt.tz is None:
            earnings_ts = earnings_ts.dt.tz_localize("UTC")
        else:
            earnings_ts = earnings_ts.dt.tz_convert("UTC")
            
        earnings_lookup = pd.DataFrame({"earnings_ts": earnings_ts})
        earnings_lookup = earnings_lookup.sort_values("earnings_ts")
        
        # Use merge_asof to find the next earnings date
        # direction='forward' finds the first row in right where right_on >= left_on
        merged = pd.merge_asof(
            out,
            earnings_lookup,
            left_on="ts",
            right_on="earnings_ts",
            direction="forward"
        )
        
        # Calculate difference in days
        diff = merged["earnings_ts"] - merged["ts"]
        out["days_until_earnings"] = diff.dt.total_seconds() / 86400.0
    else:
        out["days_until_earnings"] = np.nan

    # Carry forward VWAP; ensure non-null if present
    out["vwap"] = out["vwap"].ffill()

    # Merge VIXY Context if available
    vix_ctx = opts.get("vixy_context")
    if vix_ctx is not None and not vix_ctx.empty:
        # We need to merge on TS. 
        # Ideally an asof merge if timestamps strictly aligned is risky, but inner join might drop data.
        # Given both are 1-min bars from same source, left join is safest.
        # Ensure dtypes match
        if out["ts"].dt.tz is None: out["ts"] = out["ts"].dt.tz_localize("UTC")
        if vix_ctx["ts"].dt.tz is None: vix_ctx["ts"] = vix_ctx["ts"].dt.tz_localize("UTC")
            
        out = pd.merge(out, vix_ctx, on="ts", how="left")
        
        # Forward fill VIX metrics to handle gaps
        cols = ["vix_log_ret", "vix_z_score", "vix_rel_vol", "vix_atr_ratio"]
        out[cols] = out[cols].ffill().fillna(0.0)
    else:
        out["vix_log_ret"] = 0.0
        out["vix_z_score"] = 0.0
        out["vix_rel_vol"] = 0.0
        out["vix_atr_ratio"] = 0.0

    # Round all float columns to 5 decimal places
    float_cols = out.select_dtypes(include=['float64', 'float32']).columns
    out[float_cols] = out[float_cols].round(5)

    return out


def engineer_features(df: pd.DataFrame, options: dict | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    
    opts = options or {}
    enable_segmentation = opts.get("enable_segmentation", False)
    train_window = int(opts.get("train_window", 30))
    test_window = int(opts.get("test_window", 5))
    segment_size = train_window + test_window
    
    # 1. Calculate Features Globally (Preserves History for SMAs etc)
    out = _calculate_features_for_chunk(df, opts)

    if enable_segmentation and segment_size > 0:
        # 2. THEN Create Segments / Splits
        out = out.sort_values("ts")
        
        # Vectorized assignment of split labels
        # Create an array of indices [0, 1, 2, ...]
        indices = np.arange(len(out))
        # Modulo by segment size to get position within segment [0..35]
        positions = indices % segment_size
        
        # Identify which positions are 'test'
        # Train: 0 to train_window-1
        # Test: train_window to end
        is_test = positions >= train_window
        
        out["data_split"] = np.where(is_test, "test", "train")
    else:
        out["data_split"] = "train"

    return out

def ensure_dest_schema(dest_conn: duckdb.DuckDBPyConnection) -> None:
    # Check if we need to migrate schema (add options to PK)
    try:
        dest_conn.execute("SELECT options FROM feature_bars LIMIT 0")
    except duckdb.Error:
        # If table exists but options column is missing, drop it to recreate with new PK
        if _has_table(dest_conn, "feature_bars"):
            logger.warning("Dropping feature_bars to update schema with options in PK")
            dest_conn.execute("DROP TABLE feature_bars")

    # Migration for new columns
    for col in ["volume_change", "lag_1_close", "dist_vwap", "intraday_intensity", "vol_adj_mom_20", 
                "log_return_1m", "return_z_score_20", "vol_ratio_60", "atr_ratio_15",
                "vix_log_ret", "vix_z_score", "vix_rel_vol", "vix_atr_ratio"]:
        try:
             dest_conn.execute(f"SELECT {col} FROM feature_bars LIMIT 0")
        except duckdb.Error:
             if _has_table(dest_conn, "feature_bars"):
                 logger.warning(f"Adding missing column {col} to feature_bars")
                 dest_conn.execute(f"ALTER TABLE feature_bars ADD COLUMN {col} DOUBLE")

    dest_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_bars (
            symbol VARCHAR,
            ts TIMESTAMP WITH TIME ZONE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            vwap DOUBLE,
            trade_count BIGINT,
            return_1m DOUBLE,
            lag_1_close DOUBLE,
            dist_vwap DOUBLE,
            intraday_intensity DOUBLE,
            vol_adj_mom_20 DOUBLE,
            log_return_1m DOUBLE,
            return_z_score_20 DOUBLE,
            vol_ratio_60 DOUBLE,
            atr_ratio_15 DOUBLE,
            vix_log_ret DOUBLE,
            vix_z_score DOUBLE,
            vix_rel_vol DOUBLE,
            vix_atr_ratio DOUBLE,
            sma_close_5 DOUBLE,
            sma_close_20 DOUBLE,
            bb_upper_20_2 DOUBLE,
            bb_lower_20_2 DOUBLE,
            rsi_14 DOUBLE,
            macd_line DOUBLE,
            macd_signal DOUBLE,
            macd_hist DOUBLE,
            atr_14 DOUBLE,
            vol_sma_20 DOUBLE,
            volume_change DOUBLE,
            time_of_day_min INTEGER,
            day_of_week INTEGER,
            day_of_month INTEGER,
            month INTEGER,
            days_until_earnings DOUBLE,
            data_split VARCHAR,
            options VARCHAR,
            PRIMARY KEY (symbol, ts, options)
        );
        """
    )


def ensure_metadata_schema(dest_conn: duckdb.DuckDBPyConnection) -> None:
    dest_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_runs (
            run_id VARCHAR,
            ts TIMESTAMP WITH TIME ZONE,
            symbols VARCHAR,
            options VARCHAR,
            inserted_rows BIGINT,
            PRIMARY KEY (run_id)
        );
        """
    )


def log_run_metadata(
    dest_conn: duckdb.DuckDBPyConnection,
    symbols_arg: Iterable[str] | None,
    options: dict | None,
    inserted_rows: int
) -> None:
    ensure_metadata_schema(dest_conn)
    
    run_id = str(uuid.uuid4())
    ts = datetime.now(tz=timezone.utc)
    
    if symbols_arg is None:
        symbols_str = "ALL"
    else:
        s_list = list(symbols_arg)
        if len(s_list) > 20:
             symbols_str = f"{','.join(s_list[:20])},... ({len(s_list)} total)"
        else:
             symbols_str = ",".join(s_list)

    options_json = json.dumps(options or {})
    
    dest_conn.execute(
        "INSERT INTO feature_runs (run_id, ts, symbols, options, inserted_rows) VALUES (?, ?, ?, ?, ?)",
        [run_id, ts, symbols_str, options_json, inserted_rows]
    )


def write_features(
    dest_conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    symbol: str,
    parquet_root: Path,
    options: dict | None = None,
) -> int:
    if df.empty:
        return 0
    ensure_dest_schema(dest_conn)
    
    # Ensure symbol column exists
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    # Add options column for uniqueness
    opts_str = json.dumps(options or {}, sort_keys=True)
    df["options"] = opts_str

    dest_conn.register("features_df", df)
    dest_conn.execute("INSERT OR REPLACE INTO feature_bars BY NAME SELECT * FROM features_df")
    dest_conn.unregister("features_df")

    # Write partitioned parquet for downstream consumption
    df["date"] = pd.to_datetime(df["ts"]).dt.date
    current_hash = hashlib.md5(opts_str.encode()).hexdigest()[:8]

    for date, group in df.groupby("date"):
        dest = parquet_root / symbol / f"dt={date}"
        dest.mkdir(parents=True, exist_ok=True)
        
        # 1. Legacy Migration: Rename unhashed features.parquet if it exists
        legacy_path = dest / "features.parquet"
        if legacy_path.exists():
            try:
                # Read options column to determine hash
                legacy_df = pd.read_parquet(legacy_path, columns=["options"])
                if not legacy_df.empty:
                    old_opt = legacy_df["options"].iloc[0]
                    if pd.isna(old_opt) or not old_opt: old_opt = "{}"
                    old_hash = hashlib.md5(old_opt.encode()).hexdigest()[:8]
                    
                    target_name = dest / f"features_{old_hash}.parquet"
                    if not target_name.exists():
                        legacy_path.rename(target_name)
                    else:
                        legacy_path.unlink() # Duplicate
            except Exception as e:
                logger.warning(f"Legacy migration failed: {e}")

        file_path = dest / f"features_{current_hash}.parquet"
        group.drop(columns=["date"]).to_parquet(file_path, index=False)
    return len(df)


def run_pipeline(
    source_db: Path,
    dest_db: Path,
    dest_parquet: Path,
    symbols: Iterable[str] | None = None,
    options: dict | None = None,
    progress_callback: callable = None,
    stop_event: threading.Event = None
) -> dict:
    if not source_db.exists():
        logger.error("source db missing", extra={"path": str(source_db)})
        return {"symbols": 0, "inserted": 0, "error": "source db missing"}

    tmpdir = tempfile.TemporaryDirectory()
    tmp_src = Path(tmpdir.name) / source_db.name
    shutil.copy2(source_db, tmp_src)
    wal_src = source_db.with_suffix(source_db.suffix + ".wal")
    if wal_src.exists():
        wal_dst = tmp_src.with_suffix(tmp_src.suffix + ".wal")
        shutil.copy2(wal_src, wal_dst)
    src_conn = duckdb.connect(str(tmp_src), read_only=True)
    dest_conn = duckdb.connect(str(dest_db))
    try:
        if not _has_table(src_conn, "bars"):
            logger.warning("source bars table missing; nothing to process", extra={"source_db": str(source_db)})
            return {"symbols": 0, "inserted": 0}
        symbol_list = list(symbols) if symbols is not None else list_symbols(src_conn)
        totals_inserted = 0
        
        total_symbols = len(symbol_list)
        
        # Pre-fetch VIXY data for market context
        vixy_ctx = pd.DataFrame()
        try:
            vixy_df = fetch_bars(src_conn, "VIXY")
            if not vixy_df.empty:
                logger.info("Fetched VIXY data for market context context", extra={"rows": len(vixy_df)})
                vixy_clean = clean_bars(vixy_df)
                vixy_ctx = _calculate_vix_metrics(vixy_clean)
            else:
                 logger.warning("VIXY data not found. VIX features will be zero.")
        except Exception as e:
            logger.error(f"Failed to calculate VIXY metrics: {e}")

        for i, sym in enumerate(symbol_list):
            # Check Stop Signal
            if stop_event and stop_event.is_set():
                logger.warning("Pipeline stopped by user request")
                break
                
            # Report Progress
            if progress_callback:
                progress_callback(sym, i + 1, total_symbols, "Fetching bars")

            df = fetch_bars(src_conn, sym)
            if df.empty:
                logger.info("no source bars", extra={"symbol": sym})
                continue
            
            # Fetch earnings and inject into options for calculation
            earnings_df = fetch_earnings(src_conn, sym)
            calc_options = (options or {}).copy()
            calc_options["earnings_df"] = earnings_df
            calc_options["vixy_context"] = vixy_ctx

            if progress_callback:
                progress_callback(sym, i + 1, total_symbols, "Engineering features")

            cleaned = clean_bars(df)
            featured = engineer_features(cleaned, calc_options)
            
            if progress_callback:
                progress_callback(sym, i + 1, total_symbols, "Writing output")

            # Pass original options to write_features (excludes earnings_df)
            inserted = write_features(dest_conn, featured, sym, dest_parquet, options)
            totals_inserted += inserted
            logger.info(
                "feature_build_complete",
                extra={
                    "symbol": sym,
                    "source_rows": len(df),
                    "clean_rows": len(cleaned),
                    "feature_rows": len(featured),
                    "inserted": inserted,
                },
            )
        
        log_run_metadata(dest_conn, symbols, options, totals_inserted)
        return {"symbols": len(symbol_list), "inserted": totals_inserted}
    finally:
        try:
            dest_conn.close()
        finally:
            src_conn.close()
            tmpdir.cleanup()

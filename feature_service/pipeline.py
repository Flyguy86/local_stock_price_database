from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import tempfile
import shutil

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
    else:
        out["vol_sma_20"] = np.nan

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

    # Placeholder for future enrichment; set NaN for now
    out["days_until_earnings"] = pd.Series(np.nan, index=out.index, dtype="float64")

    # Carry forward VWAP; ensure non-null if present
    out["vwap"] = out["vwap"].ffill()

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
    
    if enable_segmentation and segment_size > 0:
        # Create segments
        df = df.copy()
        df = df.sort_values("ts")
        df["_seg_id"] = np.arange(len(df)) // segment_size
        
        results = []
        for seg_id, group in df.groupby("_seg_id"):
            # Calculate features for this group
            res = _calculate_features_for_chunk(group, opts)
            
            # Add split label
            n = len(res)
            splits = ["train"] * min(n, train_window) + ["test"] * max(0, n - train_window)
            res["data_split"] = splits[:n]
            results.append(res)
            
        out = pd.concat(results)
        out = out.drop(columns=["_seg_id"], errors="ignore")
    else:
        out = _calculate_features_for_chunk(df, opts)
        out["data_split"] = "train"

    return out


def ensure_dest_schema(dest_conn: duckdb.DuckDBPyConnection) -> None:
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
            time_of_day_min INTEGER,
            day_of_week INTEGER,
            day_of_month INTEGER,
            month INTEGER,
            days_until_earnings DOUBLE,
            data_split VARCHAR,
            PRIMARY KEY (symbol, ts)
        );
        """
    )
    try:
        dest_conn.execute("ALTER TABLE feature_bars ADD COLUMN IF NOT EXISTS data_split VARCHAR")
    except duckdb.Error:
        pass


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
) -> int:
    if df.empty:
        return 0
    ensure_dest_schema(dest_conn)
    
    # Ensure symbol column exists
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    dest_conn.register("features_df", df)
    dest_conn.execute("INSERT OR REPLACE INTO feature_bars BY NAME SELECT * FROM features_df")
    dest_conn.unregister("features_df")

    # Write partitioned parquet for downstream consumption
    df["date"] = pd.to_datetime(df["ts"]).dt.date
    for date, group in df.groupby("date"):
        dest = parquet_root / symbol / f"dt={date}"
        dest.mkdir(parents=True, exist_ok=True)
        file_path = dest / "features.parquet"
        group.drop(columns=["date"]).to_parquet(file_path, index=False)
    return len(df)


def run_pipeline(
    source_db: Path,
    dest_db: Path,
    dest_parquet: Path,
    symbols: Iterable[str] | None = None,
    options: dict | None = None,
) -> dict:
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

        for sym in symbol_list:
            df = fetch_bars(src_conn, sym)
            if df.empty:
                logger.info("no source bars", extra={"symbol": sym})
                continue
            cleaned = clean_bars(df)
            featured = engineer_features(cleaned, options)
            inserted = write_features(dest_conn, featured, sym, dest_parquet)
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

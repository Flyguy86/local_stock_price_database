from __future__ import annotations
import logging
import pandas as pd
import duckdb
import tempfile
import shutil
from datetime import time
from pathlib import Path

log = logging.getLogger("app.backfill")


class BackfillManager:
    """Handles detection and filling of missing 1-minute bar data"""
    
    # US Stock Market hours: 9:30 AM - 4:00 PM ET
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    
    def __init__(self, db_path: Path, parquet_root: Path):
        self.db_path = db_path
        self.parquet_root = parquet_root
    
    def _is_market_hours(self, ts: pd.Timestamp) -> bool:
        """Check if timestamp falls within market trading hours"""
        # Convert to US/Eastern timezone
        et_ts = ts.tz_convert('US/Eastern')
        
        # Check if weekday (Monday=0, Sunday=6)
        if et_ts.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check if within market hours (9:30 AM - 4:00 PM ET)
        ts_time = et_ts.time()
        return self.MARKET_OPEN <= ts_time < self.MARKET_CLOSE
    
    def find_missing_bars(self, symbol: str, limit: int = 1) -> pd.DataFrame:
        """
        Find missing 1-minute bars for a symbol during market hours.
        
        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of gaps to return (default: 1 for iterative processing)
        
        Returns:
            DataFrame with columns: symbol, expected_ts, prev_ts, next_ts
        """
        # Open a new read-only connection (safe since we're only reading)
        conn = duckdb.connect(str(self.db_path), read_only=True)
        
        try:
            # Get all bars for symbol ordered by timestamp
            query = """
                SELECT ts, open, high, low, close, volume, vwap, trade_count
                FROM bars 
                WHERE symbol = ?
                ORDER BY ts
            """
            df = conn.execute(query, [symbol]).fetch_df()
            
            if df.empty or len(df) < 2:
                log.info("insufficient data for gap detection", extra={"symbol": symbol, "rows": len(df)})
                return pd.DataFrame()
            
            # Convert to timezone-aware timestamps
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
            
            # Find gaps greater than 1 minute
            gaps = []
            for i in range(len(df) - 1):
                current_ts = df.iloc[i]['ts']
                next_ts = df.iloc[i + 1]['ts']
                
                # Expected next timestamp (1 minute later)
                expected_ts = current_ts + pd.Timedelta(minutes=1)
                
                # Check if there's a gap
                while expected_ts < next_ts:
                    # Only consider gaps during market hours
                    if self._is_market_hours(expected_ts):
                        gaps.append({
                            'symbol': symbol,
                            'expected_ts': expected_ts,
                            'prev_ts': current_ts,
                            'next_ts': next_ts,
                            'prev_idx': i,
                            'next_idx': i + 1
                        })
                        
                        if len(gaps) >= limit:
                            break
                    
                    expected_ts += pd.Timedelta(minutes=1)
                
                if len(gaps) >= limit:
                    break
            
            result_df = pd.DataFrame(gaps)
            log.info("gap detection complete", extra={
                "symbol": symbol,
                "gaps_found": len(result_df),
                "limit": limit
            })
            return result_df
            
        finally:
            conn.close()
    
    def fill_missing_bar(self, symbol: str, expected_ts: pd.Timestamp, prev_ts: pd.Timestamp, next_ts: pd.Timestamp) -> int:
        """
        Fill a single missing bar with the mean of adjacent bars.
        
        Args:
            symbol: Stock ticker symbol
            expected_ts: Timestamp of the missing bar
            prev_ts: Timestamp of the previous bar
            next_ts: Timestamp of the next bar
        
        Returns:
            Number of rows inserted (should be 1)
        """
        conn = duckdb.connect(str(self.db_path))
        
        try:
            # Get the previous and next bars
            query = """
                SELECT ts, open, high, low, close, volume, vwap, trade_count
                FROM bars 
                WHERE symbol = ? AND ts IN (?, ?)
                ORDER BY ts
            """
            adjacent_bars = conn.execute(query, [symbol, prev_ts, next_ts]).fetch_df()
            
            if len(adjacent_bars) != 2:
                log.warning("could not find adjacent bars", extra={
                    "symbol": symbol,
                    "expected_ts": expected_ts.isoformat(),
                    "found_rows": len(adjacent_bars)
                })
                return 0
            
            # Calculate mean values (excluding timestamp)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
            mean_values = adjacent_bars[numeric_cols].mean()
            
            # Create the new bar
            new_bar = pd.DataFrame([{
                'symbol': symbol,
                'ts': expected_ts,
                'open': mean_values['open'],
                'high': mean_values['high'],
                'low': mean_values['low'],
                'close': mean_values['close'],
                'volume': mean_values['volume'],
                'vwap': mean_values['vwap'],
                'trade_count': int(mean_values['trade_count']),
                'source': 'backfill'
            }])
            
            # Insert into database
            conn.register("tmp_backfill", new_bar)
            conn.execute("""
                INSERT INTO bars
                SELECT * FROM tmp_backfill
                ON CONFLICT DO NOTHING
            """)
            conn.unregister("tmp_backfill")
            
            # Also update parquet file
            new_bar['date'] = expected_ts.date()
            dest = self.parquet_root / symbol / f"dt={new_bar['date'].iloc[0]}"
            dest.mkdir(parents=True, exist_ok=True)
            file_path = dest / "bars.parquet"
            
            if file_path.exists():
                existing = pd.read_parquet(file_path)
                merged = (
                    pd.concat([existing, new_bar.drop(columns=['date'])], ignore_index=True)
                    .drop_duplicates(subset=['ts'])
                    .sort_values('ts')
                )
            else:
                merged = new_bar.drop(columns=['date']).sort_values('ts')
            
            merged.to_parquet(file_path, index=False)
            
            log.info("backfilled missing bar", extra={
                "symbol": symbol,
                "ts": expected_ts.isoformat(),
                "open": float(mean_values['open']),
                "close": float(mean_values['close'])
            })
            
            return 1
            
        finally:
            conn.close()
    
    def backfill_symbol(self, symbol: str, max_iterations: int = 100) -> dict:
        """
        Iteratively backfill missing bars for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            max_iterations: Maximum number of gaps to fill in one run
        
        Returns:
            Dict with statistics: filled count, remaining gaps, etc.
        """
        filled_count = 0
        
        for i in range(max_iterations):
            # Find one gap at a time
            gaps_df = self.find_missing_bars(symbol, limit=1)
            
            if gaps_df.empty:
                log.info("backfill complete - no more gaps", extra={
                    "symbol": symbol,
                    "filled_count": filled_count
                })
                break
            
            # Fill the first gap
            gap = gaps_df.iloc[0]
            result = self.fill_missing_bar(
                symbol,
                gap['expected_ts'],
                gap['prev_ts'],
                gap['next_ts']
            )
            
            if result > 0:
                filled_count += 1
            else:
                log.warning("failed to fill gap, stopping", extra={
                    "symbol": symbol,
                    "iteration": i,
                    "filled_count": filled_count
                })
                break
        
        return {
            "symbol": symbol,
            "filled": filled_count,
            "iterations": min(i + 1, max_iterations)
        }

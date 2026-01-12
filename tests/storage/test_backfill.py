from pathlib import Path
import pandas as pd
import pytest
from app.storage.duckdb_client import DuckDBClient
from app.storage.backfill import BackfillManager


def create_bars_with_gap(symbol: str = "TEST") -> pd.DataFrame:
    """Create sample 1-minute bars with a gap during market hours"""
    # Create bars at 9:30, 9:31 AM ET (market open), skip 9:32, then 9:33
    # 2024-01-02 is a Tuesday (market day)
    timestamps = pd.to_datetime([
        "2024-01-02T14:30:00Z",  # 9:30 AM ET
        "2024-01-02T14:31:00Z",  # 9:31 AM ET
        # Missing: 9:32 AM ET (14:32:00Z)
        "2024-01-02T14:33:00Z",  # 9:33 AM ET
    ], utc=True)
    
    return pd.DataFrame({
        "ts": timestamps,
        "open": [100.0, 101.0, 103.0],
        "high": [102.0, 103.0, 105.0],
        "low": [99.0, 100.0, 102.0],
        "close": [101.0, 102.0, 104.0],
        "volume": [1000, 1100, 1200],
        "vwap": [100.5, 101.5, 103.5],
        "trade_count": [10, 11, 12],
    })


def test_backfill_finds_gap(tmp_path: Path):
    """Test that backfill can detect a missing bar during market hours"""
    db_path = tmp_path / "test.duckdb"
    parquet_root = tmp_path / "parquet"
    
    # Setup
    db_client = DuckDBClient(db_path, parquet_root)
    df = create_bars_with_gap()
    db_client.insert_bars(df, "TEST", source="test")
    # Close connection to allow read-only access
    db_client.conn.close()
    
    # Test gap detection
    backfill_mgr = BackfillManager(db_path, parquet_root)
    gaps = backfill_mgr.find_missing_bars("TEST", limit=10)
    
    # Should find one gap at 9:32 AM ET (14:32 UTC)
    assert len(gaps) == 1
    assert gaps.iloc[0]['symbol'] == "TEST"
    
    expected_ts = pd.Timestamp("2024-01-02T14:32:00Z", tz='UTC')
    assert gaps.iloc[0]['expected_ts'] == expected_ts


def test_backfill_fills_gap_with_mean(tmp_path: Path):
    """Test that backfill fills missing bar with mean of adjacent bars"""
    db_path = tmp_path / "test.duckdb"
    parquet_root = tmp_path / "parquet"
    
    # Setup
    db_client = DuckDBClient(db_path, parquet_root)
    df = create_bars_with_gap()
    db_client.insert_bars(df, "TEST", source="test")
    db_client.conn.close()
    
    # Backfill the gap
    backfill_mgr = BackfillManager(db_path, parquet_root)
    gaps = backfill_mgr.find_missing_bars("TEST", limit=1)
    assert len(gaps) == 1
    
    gap = gaps.iloc[0]
    filled = backfill_mgr.fill_missing_bar(
        gap['symbol'],
        gap['expected_ts'],
        gap['prev_ts'],
        gap['next_ts']
    )
    
    assert filled == 1
    
    # Verify the filled bar has mean values
    db_client2 = DuckDBClient(db_path, parquet_root)
    all_bars = db_client2.latest_bars("TEST", limit=10)
    db_client2.conn.close()
    
    # Sort by timestamp to see all bars
    all_bars = all_bars.sort_values('ts')
    
    # Should have 4 bars total (3 original + 1 filled)
    assert len(all_bars) == 4
    
    # Check the filled bar (should be at index 2 when sorted by ts)
    filled_bar = all_bars.iloc[2]
    
    # Original data: prev bar has open=101, close=102; next bar has open=103, close=104
    # Mean of (101, 103) for open should be 102
    assert filled_bar['open'] == pytest.approx(102.0)
    # Mean of (102, 104) for close should be 103
    assert filled_bar['close'] == pytest.approx(103.0)
    # Source should be marked as 'backfill'
    assert filled_bar['source'] == 'backfill'


def test_backfill_symbol_iterative(tmp_path: Path):
    """Test iterative backfill of multiple gaps"""
    db_path = tmp_path / "test.duckdb"
    parquet_root = tmp_path / "parquet"
    
    # Create bars with multiple gaps
    timestamps = pd.to_datetime([
        "2024-01-02T14:30:00Z",  # 9:30 AM ET
        "2024-01-02T14:31:00Z",  # 9:31 AM ET
        # Missing: 9:32 AM ET
        # Missing: 9:33 AM ET
        "2024-01-02T14:34:00Z",  # 9:34 AM ET
    ], utc=True)
    
    df = pd.DataFrame({
        "ts": timestamps,
        "open": [100.0, 101.0, 104.0],
        "high": [102.0, 103.0, 106.0],
        "low": [99.0, 100.0, 103.0],
        "close": [101.0, 102.0, 105.0],
        "volume": [1000, 1100, 1300],
        "vwap": [100.5, 101.5, 104.5],
        "trade_count": [10, 11, 13],
    })
    
    # Setup
    db_client = DuckDBClient(db_path, parquet_root)
    db_client.insert_bars(df, "TEST", source="test")
    db_client.conn.close()
    
    # Backfill all gaps
    backfill_mgr = BackfillManager(db_path, parquet_root)
    result = backfill_mgr.backfill_symbol("TEST", max_iterations=10)
    
    # Should have filled 2 gaps
    assert result['filled'] == 2
    assert result['symbol'] == "TEST"
    
    # Verify all bars are now present
    db_client2 = DuckDBClient(db_path, parquet_root)
    all_bars = db_client2.latest_bars("TEST", limit=10)
    db_client2.conn.close()
    assert len(all_bars) == 5  # Original 3 + 2 filled


def test_backfill_ignores_weekend_gaps(tmp_path: Path):
    """Test that backfill ignores gaps on weekends"""
    db_path = tmp_path / "test.duckdb"
    parquet_root = tmp_path / "parquet"
    
    # Create bars on Friday and Monday (skip Saturday/Sunday)
    # 2024-01-05 is Friday, 2024-01-08 is Monday
    timestamps = pd.to_datetime([
        "2024-01-05T20:00:00Z",  # Friday 3:00 PM ET (near close)
        "2024-01-08T14:30:00Z",  # Monday 9:30 AM ET (market open)
    ], utc=True)
    
    df = pd.DataFrame({
        "ts": timestamps,
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1100],
        "vwap": [100.5, 101.5],
        "trade_count": [10, 11],
    })
    
    # Setup
    db_client = DuckDBClient(db_path, parquet_root)
    db_client.insert_bars(df, "TEST", source="test")
    db_client.conn.close()
    
    # Try to find gaps
    backfill_mgr = BackfillManager(db_path, parquet_root)
    gaps = backfill_mgr.find_missing_bars("TEST", limit=100)
    
    # Should find many gaps but they should all be during market hours on weekdays
    # Weekend gaps should be excluded
    for _, gap in gaps.iterrows():
        ts = gap['expected_ts']
        assert backfill_mgr._is_market_hours(ts), f"Gap found outside market hours: {ts}"


def test_backfill_no_gaps(tmp_path: Path):
    """Test backfill when there are no gaps"""
    db_path = tmp_path / "test.duckdb"
    parquet_root = tmp_path / "parquet"
    
    # Create continuous bars
    timestamps = pd.to_datetime([
        "2024-01-02T14:30:00Z",
        "2024-01-02T14:31:00Z",
        "2024-01-02T14:32:00Z",
    ], utc=True)
    
    df = pd.DataFrame({
        "ts": timestamps,
        "open": [100.0, 101.0, 102.0],
        "high": [102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [1000, 1100, 1200],
        "vwap": [100.5, 101.5, 102.5],
        "trade_count": [10, 11, 12],
    })
    
    # Setup
    db_client = DuckDBClient(db_path, parquet_root)
    db_client.insert_bars(df, "TEST", source="test")
    db_client.conn.close()
    
    # Try backfill
    backfill_mgr = BackfillManager(db_path, parquet_root)
    result = backfill_mgr.backfill_symbol("TEST", max_iterations=10)
    
    # Should fill 0 gaps
    assert result['filled'] == 0

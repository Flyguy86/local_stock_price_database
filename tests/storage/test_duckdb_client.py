from pathlib import Path

import pandas as pd
import pytest

from app.storage.duckdb_client import DuckDBClient


def make_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime([
                "2024-01-01T12:00:00Z",
                "2024-01-01T12:01:00Z",
                "2024-01-01T12:01:00Z",
            ], utc=True),
            "open": [1.0, 1.1, 1.1],
            "high": [1.2, 1.3, 1.3],
            "low": [0.9, 1.0, 1.0],
            "close": [1.05, 1.15, 1.15],
            "volume": [100, 150, 150],
            "vwap": [1.02, 1.12, 1.12],
            "trade_count": [5, 6, 6],
        }
    )


@pytest.mark.parametrize("limit", [1, 2])
def test_duckdb_insert_and_read(tmp_path: Path, limit: int):
    db_path = tmp_path / "test.duckdb"
    parquet_root = tmp_path / "parquet"
    client = DuckDBClient(db_path, parquet_root)
    try:
        df = make_sample_frame()
        inserted = client.insert_bars(df, "AAPL", source="alpaca")
        assert inserted == 2

        latest = client.latest_bars("AAPL", limit=limit)
        assert 0 < len(latest) <= limit
        assert latest.iloc[0]["symbol"] == "AAPL"

        symbols = client.list_symbols()
        assert symbols == ["AAPL"]

        page_df, total = client.bars_page("AAPL", limit=10, offset=0)
        assert total == 2
        assert len(page_df) == 2

        stored_partition = parquet_root / "AAPL" / "dt=2024-01-01"
        stored_file = stored_partition / "bars.parquet"
        assert stored_file.exists()
        stored_df = pd.read_parquet(stored_file)
        assert len(stored_df) == 2

        page_df_table, total_table = client.table_page("bars", limit=10, offset=0)
        assert total_table == 2
        assert "source" in page_df_table.columns

        second_insert = client.insert_bars(df, "AAPL", source="alpaca")
        assert second_insert == 2
        after_second_insert, _ = client.bars_page("AAPL", limit=10, offset=0)
        assert len(after_second_insert) == 2
    finally:
        client.conn.close()

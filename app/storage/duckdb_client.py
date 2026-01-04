from __future__ import annotations
import duckdb
from pathlib import Path
from typing import Iterable
import pandas as pd
from .schema import table_blueprints

class DuckDBClient:
    def __init__(self, db_path: Path, parquet_root: Path):
        self.db_path = db_path
        self.parquet_root = parquet_root
        self.conn = duckdb.connect(str(self.db_path))
        for ddl in table_blueprints():
            self.conn.execute(ddl)

    def insert_bars(self, df: pd.DataFrame, symbol: str) -> int:
        if df.empty:
            return 0
        df = df.copy()
        df["symbol"] = symbol
        # Idempotent insert: skip existing (symbol, ts)
        self.conn.execute(
            """
            INSERT INTO bars
            SELECT * FROM df
            ON CONFLICT DO NOTHING
            """,
            {"df": df},
        )
        # Partitioned Parquet append
        df["date"] = df["ts"].dt.date
        for date, group in df.groupby("date"):
            dest = self.parquet_root / symbol / f"dt={date}"
            dest.mkdir(parents=True, exist_ok=True)
            file_path = dest / "bars.parquet"
            group.drop(columns=["date"]).to_parquet(file_path, index=False)
        return len(df)

    def latest_bars(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        return self.conn.execute(
            """
            SELECT * FROM bars WHERE symbol = ? ORDER BY ts DESC LIMIT ?
            """,
            [symbol, limit],
        ).fetch_df()

    def list_symbols(self) -> list[str]:
        return [
            row[0]
            for row in self.conn.execute(
                "SELECT DISTINCT symbol FROM bars ORDER BY symbol"
            ).fetchall()
        ]

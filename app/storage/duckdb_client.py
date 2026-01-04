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

    def insert_bars(self, df: pd.DataFrame, symbol: str, source: str | None = None) -> int:
        if df.empty:
            return 0
        df = df.copy()
        df["symbol"] = symbol
        df["source"] = source
        cols = ["symbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trade_count", "source"]
        df = df[cols]
        self.conn.register("tmp_df", df)
        self.conn.execute(
            """
            INSERT INTO bars
            SELECT * FROM tmp_df
            ON CONFLICT DO NOTHING
            """
        )
        self.conn.unregister("tmp_df")
        df["date"] = pd.to_datetime(df["ts"]).dt.date
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

    def list_tables(self) -> list[str]:
        return [
            row[0]
            for row in self.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
            ).fetchall()
        ]

    def bars_page(self, symbol: str, limit: int, offset: int) -> tuple[pd.DataFrame, int]:
        total = self.conn.execute(
            "SELECT COUNT(*) FROM bars WHERE symbol = ?", [symbol]
        ).fetchone()[0]
        df = self.conn.execute(
            """
            SELECT * FROM bars
            WHERE symbol = ?
            ORDER BY ts DESC
            LIMIT ? OFFSET ?
            """,
            [symbol, limit, offset],
        ).fetch_df()
        return df, total

    def table_page(self, table: str, limit: int, offset: int) -> tuple[pd.DataFrame, int]:
        if table not in self.list_tables():
            raise ValueError("table not found")
        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        ident = f'"{table}"'
        total = self.conn.execute(f"SELECT COUNT(*) FROM {ident}").fetchone()[0]
        df = self.conn.execute(
            f"SELECT * FROM {ident} LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetch_df()
        return df, total

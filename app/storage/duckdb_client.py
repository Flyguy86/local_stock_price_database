from __future__ import annotations
import duckdb
from pathlib import Path
import pandas as pd
import os
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
        df = df[cols].drop_duplicates(subset=["symbol", "ts"])
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
            if file_path.exists():
                existing = pd.read_parquet(file_path)
                merged = (
                    pd.concat([existing, group.drop(columns=["date"])], ignore_index=True)
                    .drop_duplicates(subset=["ts"])
                    .sort_values("ts")
                )
            else:
                merged = group.drop(columns=["date"]).sort_values("ts")
            merged.to_parquet(file_path, index=False)
        return len(df)

    def insert_earnings(self, df: pd.DataFrame, symbol: str) -> int:
        if df.empty:
            return 0
        df = df.copy()
        df["symbol"] = symbol
        # Ensure columns match schema
        # symbol, announce_date, report_time, fiscal_period, fiscal_end_date, actual_eps, estimated_eps
        
        # Fill missing columns with None/NaN
        expected_cols = ["announce_date", "report_time", "fiscal_period", "fiscal_end_date", "actual_eps", "estimated_eps"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
                
        cols = ["symbol"] + expected_cols
        df = df[cols].drop_duplicates(subset=["symbol", "announce_date"])
        
        self.conn.register("tmp_earnings", df)
        self.conn.execute(
            """
            INSERT INTO earnings
            SELECT * FROM tmp_earnings
            ON CONFLICT (symbol, announce_date) DO UPDATE SET
                report_time = EXCLUDED.report_time,
                fiscal_period = EXCLUDED.fiscal_period,
                fiscal_end_date = EXCLUDED.fiscal_end_date,
                actual_eps = EXCLUDED.actual_eps,
                estimated_eps = EXCLUDED.estimated_eps
            """
        )
        self.conn.unregister("tmp_earnings")
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

    def delete_symbol(self, symbol: str) -> int:
        deleted = self.conn.execute("DELETE FROM bars WHERE symbol = ?", [symbol]).rowcount
        parquet_path = self.parquet_root / symbol
        if parquet_path.exists():
            for root, dirs, files in os.walk(parquet_path, topdown=False):
                for f in files:
                    Path(root, f).unlink()
                for d in dirs:
                    Path(root, d).rmdir()
            parquet_path.rmdir()
        return deleted

    def delete_all(self) -> None:
        self.conn.execute("DELETE FROM bars")
        if self.parquet_root.exists():
            for root, dirs, files in os.walk(self.parquet_root, topdown=False):
                for f in files:
                    Path(root, f).unlink()
                for d in dirs:
                    Path(root, d).rmdir()
            self.parquet_root.mkdir(parents=True, exist_ok=True)

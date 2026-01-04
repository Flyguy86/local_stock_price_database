from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from httpx import HTTPStatusError
from ..storage.duckdb_client import DuckDBClient
from ..config import settings
from .alpaca_client import get_alpaca_client
from .iex_client import get_iex_client
import pandas as pd

log = logging.getLogger("app.poller")

class IngestPoller:
    def __init__(self, db: DuckDBClient):
        self.db = db

    async def run_history(self, symbol: str, start: str | None = None, end: str | None = None) -> dict:
        inserted = 0
        target_start = pd.to_datetime(start, utc=True) if start else None
        current_end = end or datetime.now(timezone.utc).isoformat()
        # Prefer Alpaca if configured
        if settings.alpaca_key_id and settings.alpaca_secret_key:
            client = get_alpaca_client()
            try:
                while True:
                    got_any = False
                    chunk_earliest = None
                    async for df in client.fetch_bars(symbol, timeframe="1Min", start=start, end=current_end):
                        got_any = True
                        if df.empty:
                            continue
                        chunk_start = pd.to_datetime(df["ts"]).min().isoformat()
                        chunk_end = pd.to_datetime(df["ts"]).max().isoformat()
                        rows = len(df)
                        inserted += self.db.insert_bars(df, symbol, source="alpaca")
                        log.info(
                            "ingested chunk range=%s -> %s rows=%s symbol=%s source=%s",
                            chunk_start,
                            chunk_end,
                            rows,
                            symbol,
                            "alpaca",
                        )
                        earliest_val = pd.to_datetime(df["ts"]).min()
                        chunk_earliest = earliest_val if chunk_earliest is None else min(chunk_earliest, earliest_val)
                    if not got_any or chunk_earliest is None:
                        break
                    if target_start and chunk_earliest <= target_start:
                        break
                    current_end = (chunk_earliest - pd.Timedelta(minutes=1)).isoformat()
                return {"symbol": symbol, "inserted": inserted, "ts": datetime.now(timezone.utc).isoformat()}
            except HTTPStatusError as exc:
                log.warning("alpaca fetch failed, attempting IEX", extra={"symbol": symbol, "status": exc.response.status_code})
            finally:
                await client.aclose()
        # Fallback to IEX if token present
        if settings.iex_token:
            client = get_iex_client()
            try:
                while True:
                    got_any = False
                    chunk_earliest = None
                    async for df in client.fetch_bars(symbol):
                        got_any = True
                        if df.empty:
                            continue
                        chunk_start = pd.to_datetime(df["ts"]).min().isoformat()
                        chunk_end = pd.to_datetime(df["ts"]).max().isoformat()
                        rows = len(df)
                        inserted += self.db.insert_bars(df, symbol, source="iex")
                        log.info(
                            "ingested chunk range=%s -> %s rows=%s symbol=%s source=%s",
                            chunk_start,
                            chunk_end,
                            rows,
                            symbol,
                            "iex",
                        )
                        earliest_val = pd.to_datetime(df["ts"]).min()
                        chunk_earliest = earliest_val if chunk_earliest is None else min(chunk_earliest, earliest_val)
                    if not got_any or chunk_earliest is None:
                        break
                    if target_start and chunk_earliest <= target_start:
                        break
                    current_end = (chunk_earliest - pd.Timedelta(minutes=1)).isoformat()
                return {"symbol": symbol, "inserted": inserted, "ts": datetime.now(timezone.utc).isoformat()}
            finally:
                await client.aclose()
        raise RuntimeError("No data provider succeeded for symbol")

    async def run_live_once(self, symbol: str) -> dict:
        return await self.run_history(symbol)

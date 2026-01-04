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
        current_end = pd.to_datetime(end, utc=True) if end else pd.Timestamp.now(tz=timezone.utc)
        window = pd.Timedelta(days=180)

        async def process_frame(df: pd.DataFrame, source: str):
            nonlocal inserted
            if df.empty:
                return None
            chunk_start = pd.to_datetime(df["ts"]).min()
            chunk_end = pd.to_datetime(df["ts"]).max()
            rows = len(df)
            inserted += self.db.insert_bars(df, symbol, source=source)
            log.info(
                "ingested chunk range=%s -> %s rows=%s symbol=%s source=%s",
                chunk_start.isoformat(),
                chunk_end.isoformat(),
                rows,
                symbol,
                source,
            )
            return chunk_start

        # Alpaca first
        if settings.alpaca_key_id and settings.alpaca_secret_key:
            client = get_alpaca_client()
            try:
                while True:
                    window_start = current_end - window
                    if target_start:
                        window_start = max(window_start, target_start)
                    if window_start >= current_end:
                        break
                    got_any = False
                    async for df in client.fetch_bars(
                        symbol,
                        timeframe="1Min",
                        start=window_start.isoformat(),
                        end=current_end.isoformat(),
                    ):
                        got_any = True
                        earliest = await process_frame(df, "alpaca")
                        if earliest is not None:
                            current_end = earliest - pd.Timedelta(minutes=1)
                    if not got_any:
                        break
                    if target_start and current_end <= target_start:
                        break
                return {"symbol": symbol, "inserted": inserted, "ts": datetime.now(timezone.utc).isoformat()}
            except HTTPStatusError as exc:
                log.warning("alpaca fetch failed, attempting IEX", extra={"symbol": symbol, "status": exc.response.status_code})
            finally:
                await client.aclose()

        # IEX fallback
        if settings.iex_token:
            client = get_iex_client()
            try:
                current_end = pd.to_datetime(end, utc=True) if end else pd.Timestamp.now(tz=timezone.utc)
                while True:
                    window_start = current_end - window
                    if target_start:
                        window_start = max(window_start, target_start)
                    if window_start >= current_end:
                        break
                    got_any = False
                    async for df in client.fetch_bars(symbol):
                        got_any = True
                        earliest = await process_frame(df, "iex")
                        if earliest is not None:
                            current_end = earliest - pd.Timedelta(minutes=1)
                    if not got_any:
                        break
                    if target_start and current_end <= target_start:
                        break
                return {"symbol": symbol, "inserted": inserted, "ts": datetime.now(timezone.utc).isoformat()}
            finally:
                await client.aclose()

        raise RuntimeError("No data provider succeeded for symbol")

    async def run_live_once(self, symbol: str) -> dict:
        return await self.run_history(symbol)

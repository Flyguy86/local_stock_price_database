from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from httpx import HTTPStatusError
from ..storage.duckdb_client import DuckDBClient
from ..config import settings
from .alpaca_client import get_alpaca_client
from .iex_client import get_iex_client

log = logging.getLogger("app.poller")

class IngestPoller:
    def __init__(self, db: DuckDBClient):
        self.db = db

    async def _ingest_frames(self, symbol: str, frames) -> int:
        inserted = 0
        async for df in frames:
            inserted += self.db.insert_bars(df, symbol)
            log.info("ingested chunk", extra={"symbol": symbol, "rows": len(df)})
        return inserted

    async def run_history(self, symbol: str, start: str | None = None, end: str | None = None) -> dict:
        inserted = 0
        # Prefer Alpaca if configured
        if settings.alpaca_key_id and settings.alpaca_secret_key:
            client = get_alpaca_client()
            try:
                inserted = await self._ingest_frames(symbol, client.fetch_bars(symbol, timeframe="1Min", start=start, end=end))
                return {"symbol": symbol, "inserted": inserted, "ts": datetime.now(timezone.utc).isoformat()}
            except HTTPStatusError as exc:
                log.warning("alpaca fetch failed, attempting IEX", extra={"symbol": symbol, "status": exc.response.status_code})
            finally:
                await client.aclose()
        # Fallback to IEX if token present
        if settings.iex_token:
            client = get_iex_client()
            try:
                inserted = await self._ingest_frames(symbol, client.fetch_bars(symbol))
                return {"symbol": symbol, "inserted": inserted, "ts": datetime.now(timezone.utc).isoformat()}
            finally:
                await client.aclose()
        raise RuntimeError("No data provider succeeded for symbol")

    async def run_live_once(self, symbol: str) -> dict:
        return await self.run_history(symbol)

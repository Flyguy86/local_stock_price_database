from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from ..storage.duckdb_client import DuckDBClient
from ..config import settings
from .alpaca_client import get_alpaca_client

log = logging.getLogger("app.poller")

class IngestPoller:
    def __init__(self, db: DuckDBClient):
        self.db = db

    async def run_history(self, symbol: str, start: str | None = None, end: str | None = None) -> dict:
        client = get_alpaca_client()
        inserted = 0
        try:
            async for df in client.fetch_bars(symbol, timeframe="1Min", start=start, end=end):
                inserted += self.db.insert_bars(df, symbol)
                log.info("ingested chunk", extra={"symbol": symbol, "rows": len(df)})
        finally:
            await client.aclose()
        return {"symbol": symbol, "inserted": inserted, "ts": datetime.now(timezone.utc).isoformat()}

    async def run_live_once(self, symbol: str) -> dict:
        # Placeholder for near-real-time single poll
        return await self.run_history(symbol)

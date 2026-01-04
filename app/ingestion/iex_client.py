from __future__ import annotations
import logging
from typing import AsyncIterator
import httpx
import pandas as pd
from ..config import settings

log = logging.getLogger("app.iex")

class IEXClient:
    def __init__(self, token: str | None, base_url: str):
        self.token = token or ""
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def fetch_bars(
        self,
        symbol: str,
        chart_last: int | None = None,
    ) -> AsyncIterator[pd.DataFrame]:
        # Intraday 1-min prices
        params = {"token": self.token}
        if chart_last:
            params["chartLast"] = chart_last
        resp = await self._client.get(f"/stable/stock/{symbol}/intraday-prices", params=params)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return
        df = pd.DataFrame(rows)
        if {"date", "minute"}.issubset(df.columns):
            df["ts"] = pd.to_datetime(df["date"] + " " + df["minute"], utc=True)
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "numberOfTrades": "trade_count",
                "average": "vwap",
            }
        )
        yield df[["ts", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]

    async def aclose(self):
        await self._client.aclose()

def get_iex_client() -> IEXClient:
    return IEXClient(settings.iex_token, settings.iex_base_url)

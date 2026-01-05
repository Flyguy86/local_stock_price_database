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

    async def fetch_earnings(self, symbol: str, last: int = 4) -> pd.DataFrame:
        # IEX Earnings endpoint
        params = {"token": self.token}
        resp = await self._client.get(f"/stock/{symbol}/earnings/{last}", params=params)
        if resp.status_code == 404:
            return pd.DataFrame()
        resp.raise_for_status()
        data = resp.json()
        earnings_list = data.get("earnings", [])
        if not earnings_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(earnings_list)
        # Map IEX fields to our schema
        # IEX fields: actualEPS, consensusEPS, announceTime, fiscalPeriod, fiscalEndDate, reportTime? (announceTime serves as reportTime sometimes)
        # We need: announce_date, report_time, fiscal_period, fiscal_end_date, actual_eps, estimated_eps
        
        rename_map = {
            "fiscalPeriod": "fiscal_period",
            "fiscalEndDate": "fiscal_end_date",
            "actualEPS": "actual_eps",
            "consensusEPS": "estimated_eps",
            "EPSReportDate": "announce_date"
        }
        df = df.rename(columns=rename_map)
        
        # Ensure we have announce_date
        if "announce_date" not in df.columns:
             return pd.DataFrame()
             
        df["announce_date"] = pd.to_datetime(df["announce_date"]).dt.date
        if "fiscal_end_date" in df.columns:
            df["fiscal_end_date"] = pd.to_datetime(df["fiscal_end_date"]).dt.date
            
        # report_time is usually "bmo" (before market open) or "amc" (after market close) or specific time
        # IEX might provide it in "announceTime" ? 
        if "report_time" not in df.columns:
             df["report_time"] = None # Placeholder or extract if available
             
        columns = [c for c in ["announce_date", "report_time", "fiscal_period", "fiscal_end_date", "actual_eps", "estimated_eps"] if c in df.columns]
        return df[columns]

    async def aclose(self):
        await self._client.aclose()

def get_iex_client() -> IEXClient:
    return IEXClient(settings.iex_token, settings.iex_base_url)

from __future__ import annotations
import asyncio
import logging
import json
from typing import AsyncIterator
import httpx
import pandas as pd
from ..config import settings

log = logging.getLogger("app.alpaca")

class AlpacaClient:
    def __init__(self, key: str | None, secret: str | None, base_url: str, feed: str | None = None, trading_base_url: str | None = None):
        self.key = key
        self.secret = secret
        self.base_url = base_url
        self.feed = feed
        self.trading_base_url = trading_base_url or base_url
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "APCA-API-KEY-ID": self.key or "",
                "APCA-API-SECRET-KEY": self.secret or "",
            },
            timeout=30.0,
        )
        self._trading_client = httpx.AsyncClient(
            base_url=self.trading_base_url,
            headers={
                "APCA-API-KEY-ID": self.key or "",
                "APCA-API-SECRET-KEY": self.secret or "",
            },
            timeout=30.0,
        )

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: str | None = None,
        end: str | None = None,
        limit: int = 10000,
    ) -> AsyncIterator[pd.DataFrame]:
        next_page = None
        backoff = 1.0
        while True:
            params = {"timeframe": timeframe, "limit": limit}
            if next_page is not None:
                params["page_token"] = next_page
            if self.feed:
                params["feed"] = self.feed
            if start:
                params["start"] = start
            if end:
                params["end"] = end
            if settings.alpaca_debug_raw:
                log.info(
                    "alpaca raw request",
                    extra={"raw": json.dumps({"url": f"{self.base_url}/v2/stocks/{symbol}/bars", "params": params})},
                )
            resp = await self._client.get(f"/v2/stocks/{symbol}/bars", params=params)
            if settings.alpaca_debug_raw:
                raw_payload = resp.text
                try:
                    raw_payload = json.dumps(resp.json())
                except Exception:
                    pass
                log.info(
                    "alpaca raw response",
                    extra={
                        "raw": json.dumps(
                            {
                                "status_code": resp.status_code,
                                "headers": dict(resp.headers),
                                "body": raw_payload,
                            }
                        )
                    },
                )
            if resp.status_code >= 400:
                log.error(
                    "alpaca error response",
                    extra={
                        "status_code": resp.status_code,
                        "text": resp.text[:500],
                        "headers": dict(resp.headers),
                        "params": params,
                    },
                )
            if resp.status_code == 429:
                log.warning("rate limited; sleeping %.1fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            resp.raise_for_status()
            payload = resp.json()
            bars = payload.get("bars", [])
            if not bars:
                log.warning(
                    "alpaca empty page",
                    extra={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "feed": self.feed,
                        "start": start,
                        "end": end,
                        "limit": limit,
                        "page_token": next_page,
                        "response_keys": list(payload.keys()),
                        "raw_snippet": str(payload)[:500],
                        "headers": dict(resp.headers),
                    },
                )
                break
            df = pd.DataFrame(bars)
            if "t" in df.columns:
                df["ts"] = pd.to_datetime(df["t"], utc=True)
            yield df[["ts", "o", "h", "l", "c", "v", "vw", "n"]].rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "trade_count",
                }
            )
            next_page = payload.get("next_page_token")
            if not next_page:
                break

    async def get_clock(self) -> dict:
        resp = await self._trading_client.get("/v2/clock")
        resp.raise_for_status()
        return resp.json()

    async def aclose(self):
        await self._client.aclose()
        await self._trading_client.aclose()

def get_alpaca_client() -> AlpacaClient:
    return AlpacaClient(
        settings.alpaca_key_id,
        settings.alpaca_secret_key,
        settings.alpaca_base_url,
        settings.alpaca_feed,
        settings.alpaca_trading_base_url,
    )

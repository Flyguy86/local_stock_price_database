import json
from datetime import timezone

import pandas as pd
import pytest

from app.config import settings
from app.ingestion import poller as poller_module
from app.ingestion.alpaca_client import AlpacaClient
from app.ingestion.poller import IngestPoller


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class DummyAsyncClient:
    async def get(self, url: str, params: dict | None = None) -> DummyResponse:
        self.calls.append((url, dict(params or {})))
        return self._response

    async def aclose(self) -> None:
        self.closed = True


class StubAlpacaClient:
    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: str | None = None,
        end: str | None = None,
        limit: int = 3000,
        adjustments: str | None = "all",
    ):
        self.calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit": limit,
                "adjustments": adjustments,
            }
        )
        for frame in list(self._frames):
            yield frame
        self._frames.clear()

    async def aclose(self) -> None:
        self.closed = True


class DummyDuckDBClient:
    def insert_bars(self, df: pd.DataFrame, symbol: str, source: str) -> int:
        self.inserts.append((symbol, source, df.copy()))
        return len(df)


@pytest.mark.asyncio
async def test_fetch_bars_requests_expected_params():
    dummy_response = DummyResponse(payload)
    dummy_client = DummyAsyncClient(dummy_response)
    alpaca = AlpacaClient("key", "secret", "https://example.com", feed="iex")
    alpaca._client = dummy_client
    alpaca._trading_client = dummy_client

    frames = []
    async for frame in alpaca.fetch_bars("AAPL", start="2024-01-01T00:00:00Z", end="2024-01-02T00:00:00Z"):
        frames.append(frame)

    assert dummy_client.calls, "Expected at least one HTTP call"
    _, params = dummy_client.calls[0]
    assert params["limit"] == 3000
    assert params["adjustment"] == "all"
    assert params["feed"] == "iex"
    assert frames[0]["open"].iloc[0] == 1.0
    assert frames[0]["ts"].iloc[0] == pd.Timestamp("2024-01-01T12:00:00Z", tz=timezone.utc)


@pytest.mark.asyncio
async def test_run_history_ingests_alpaca_data(monkeypatch):
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01T12:00:00Z", "2024-01-01T12:01:00Z"], utc=True),
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100, 150],
            "vwap": [1.02, 1.12],
            "trade_count": [5, 6],
        }
    )
    stub_client = StubAlpacaClient([df])
    monkeypatch.setattr(poller_module, "get_alpaca_client", lambda: stub_client)
    dummy_db = DummyDuckDBClient()
    ingest = IngestPoller(dummy_db)

    result = await ingest.run_history("AAPL", start="2023-12-31T00:00:00Z", end="2024-01-02T00:00:00Z")

    assert result["symbol"] == "AAPL"
    assert result["inserted"] == len(df)
    assert dummy_db.inserts, "Expected bars to be inserted"
    _, source, inserted_df = dummy_db.inserts[0]
    assert source == "alpaca"
    assert len(inserted_df) == len(df)
    assert stub_client.calls[0]["limit"] == 3000
    assert stub_client.calls[0]["adjustments"] == "all"
    await stub_client.aclose()
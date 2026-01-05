import pandas as pd
import pytest

from app.ingestion.poller import IngestPoller


@pytest.mark.asyncio
async def test_run_live_once_delegates_to_history(monkeypatch):
    calls: list[tuple[str, str | None, str | None]] = []

    async def fake_run_history(self, symbol: str, start: str | None = None, end: str | None = None):
        calls.append((symbol, start, end))
        return {"symbol": symbol, "inserted": 1, "ts": "2024-01-01T00:00:00+00:00"}

    monkeypatch.setattr(IngestPoller, "run_history", fake_run_history)
    poller = IngestPoller(db=None)

    result = await poller.run_live_once("MSFT")

    assert calls == [("MSFT", None, None)]
    assert result == {"symbol": "MSFT", "inserted": 1, "ts": "2024-01-01T00:00:00+00:00"}


@pytest.mark.asyncio
async def test_run_live_batch_uses_recent_window(monkeypatch):
    fixed_now = pd.Timestamp("2024-01-01T12:00:00Z")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, tz=None: fixed_now))
    calls: list[tuple[str, str | None, str | None]] = []

    async def fake_run_history(self, symbol: str, start: str | None = None, end: str | None = None):
        calls.append((symbol, start, end))
        return {"symbol": symbol, "inserted": 1, "ts": "ignored"}

    monkeypatch.setattr(IngestPoller, "run_history", fake_run_history)
    poller = IngestPoller(db=None)

    result = await poller.run_live_batch(["AAPL", "MSFT"])

    expected_start = (fixed_now - pd.Timedelta(minutes=15)).isoformat()
    assert calls == [
        ("AAPL", expected_start, None),
        ("MSFT", expected_start, None),
    ]
    assert result["symbols"] == 2
    assert result["inserted"] == 2
    assert result["ts"]

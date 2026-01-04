from __future__ import annotations
import asyncio
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from ..config import settings
from ..logging import configure_json_logger
from ..storage.duckdb_client import DuckDBClient
from ..ingestion.poller import IngestPoller

logger = configure_json_logger(settings.log_level)
db = DuckDBClient(settings.duckdb_path, settings.parquet_dir)
poller = IngestPoller(db)
app = FastAPI(title="local_stock_price_database")

class IngestResponse(BaseModel):
    symbol: str
    inserted: int
    ts: str

class Status(BaseModel):
    symbol: str
    state: str
    last_update: str | None = None

# In-memory agent status
agent_status: dict[str, Status] = {}

@app.post("/ingest/{symbol}", response_model=IngestResponse)
async def ingest_symbol(symbol: str, background: BackgroundTasks, start: str | None = None, end: str | None = None):
    agent_status[symbol] = Status(symbol=symbol, state="queued", last_update=None)
    async def task():
        agent_status[symbol] = Status(symbol=symbol, state="running", last_update=None)
        try:
            result = await poller.run_history(symbol, start=start, end=end)
            agent_status[symbol] = Status(symbol=symbol, state="succeeded", last_update=result["ts"])
        except Exception as exc:
            logger.exception("ingest failed", extra={"symbol": symbol})
            agent_status[symbol] = Status(symbol=symbol, state="failed", last_update=None)
            raise exc
    background.add_task(asyncio.create_task, task())
    return IngestResponse(symbol=symbol, inserted=0, ts="queued")

@app.get("/status", response_model=list[Status])
async def status():
    return list(agent_status.values())

@app.get("/bars/{symbol}")
async def get_bars(symbol: str, limit: int = 100):
    if limit <= 0 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be 1..1000")
    df = db.latest_bars(symbol, limit)
    return df.to_dict(orient="records")

@app.get("/symbols")
async def symbols():
    return db.list_symbols()

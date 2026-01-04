from __future__ import annotations
import asyncio
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Local Stock DB Dashboard</title>
      <style>
        body { font-family: sans-serif; margin: 1rem; }
        section { margin-bottom: 1.5rem; }
        input, button { margin-right: 0.5rem; }
        table { border-collapse: collapse; }
        td, th { border: 1px solid #ccc; padding: 0.25rem 0.5rem; }
      </style>
    </head>
    <body>
      <h1>Local Stock Price Database</h1>
      <section>
        <h3>Ingest symbol</h3>
        <input id="symbol" placeholder="e.g. AAPL" />
        <input id="lookback-years" type="number" value="1" step="0.25" min="0.1" style="width:8ch" />
        <label for="lookback-years">lookback (years)</label>
        <button onclick="ingest()">Start ingest</button>
        <span id="ingest-result"></span>
      </section>

      <section>
        <h3>Agent status</h3>
        <button onclick="refreshStatus()">Refresh</button>
        <ul id="status-list"></ul>
      </section>

      <section>
        <h3>Latest bars</h3>
        <input id="bars-symbol" placeholder="symbol" />
        <input id="bars-limit" type="number" value="5" min="1" max="1000" />
        <button onclick="loadBars()">Load</button>
        <table id="bars-table">
          <thead></thead>
          <tbody></tbody>
        </table>
      </section>

      <script>
        function isoFromLookbackYears(years) {
          if (!years || isNaN(years) || years <= 0) return null;
          const now = new Date();
          const d = new Date(now);
          d.setFullYear(d.getFullYear() - years);
          return d.toISOString();
        }

        async function ingest() {
          const symbol = document.getElementById("symbol").value.trim();
          const startManual = document.getElementById("start").value.trim();
          const end = document.getElementById("end").value.trim();
          const lookback = parseFloat(document.getElementById("lookback-years").value);
          if (!symbol) { alert("Enter a symbol"); return; }
          const qs = new URLSearchParams();
          const startFromLookback = startManual || isoFromLookbackYears(lookback);
          if (startFromLookback) qs.append("start", startFromLookback);
          if (end) qs.append("end", end);
          const res = await fetch(`/ingest/${symbol}?` + qs.toString(), { method: "POST" });
          const data = await res.json();
          document.getElementById("ingest-result").textContent = JSON.stringify(data);
          refreshStatus();
        }

        async function refreshStatus() {
          const res = await fetch("/status");
          const data = await res.json();
          const list = document.getElementById("status-list");
          list.innerHTML = "";
          data.forEach(s => {
            const li = document.createElement("li");
            li.textContent = `${s.symbol}: ${s.state}` + (s.last_update ? ` @ ${s.last_update}` : "");
            list.appendChild(li);
          });
        }

        async function loadBars() {
          const symbol = document.getElementById("bars-symbol").value.trim();
          const limit = document.getElementById("bars-limit").value;
          if (!symbol) { alert("Enter a symbol"); return; }
          const res = await fetch(`/bars/${symbol}?limit=${limit}`);
          const data = await res.json();
          const head = document.querySelector("#bars-table thead");
          const body = document.querySelector("#bars-table tbody");
          head.innerHTML = ""; body.innerHTML = "";
          if (data.length === 0) { body.innerHTML = "<tr><td>No data</td></tr>"; return; }
          const cols = Object.keys(data[0]);
          const trHead = document.createElement("tr");
          cols.forEach(c => { const th = document.createElement("th"); th.textContent = c; trHead.appendChild(th); });
          head.appendChild(trHead);
          data.forEach(row => {
            const tr = document.createElement("tr");
            cols.forEach(c => { const td = document.createElement("td"); td.textContent = row[c]; tr.appendChild(td); });
            body.appendChild(tr);
          });
        }

        refreshStatus();
      </script>
    </body>
    </html>
    """

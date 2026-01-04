from __future__ import annotations
import asyncio
import logging
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ..config import settings
from ..logging import configure_json_logger
from ..storage.duckdb_client import DuckDBClient
from ..ingestion.poller import IngestPoller

logger = configure_json_logger(settings.log_level)
LOG_BUFFER_MAX = 200
log_buffer: list[dict] = []
log_lock = threading.Lock()

class InMemoryHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": getattr(record, "ts", None) or getattr(record, "created", None),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        with log_lock:
            log_buffer.append(payload)
            if len(log_buffer) > LOG_BUFFER_MAX:
                del log_buffer[: len(log_buffer) - LOG_BUFFER_MAX]

logger.addHandler(InMemoryHandler())
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
    error_message: str | None = None

# In-memory agent status
agent_status: dict[str, Status] = {}

@app.post("/ingest/{symbol}", response_model=IngestResponse)
async def ingest_symbol(symbol: str, start: str | None = None, end: str | None = None):
    if not ((settings.alpaca_key_id and settings.alpaca_secret_key) or settings.iex_token):
        raise HTTPException(status_code=400, detail="Missing provider credentials: set Alpaca keys or IEX token")
    logger.info("ingest request queued", extra={"symbol": symbol, "start": start, "end": end})
    agent_status[symbol] = Status(symbol=symbol, state="queued", last_update=None)
    async def task():
        agent_status[symbol] = Status(symbol=symbol, state="running", last_update=None)
        logger.info("ingest task started", extra={"symbol": symbol, "start": start, "end": end})
        try:
            result = await poller.run_history(symbol, start=start, end=end)
            agent_status[symbol] = Status(symbol=symbol, state="succeeded", last_update=result["ts"])
            logger.info("ingest task succeeded", extra={"symbol": symbol, "inserted": result["inserted"], "ts": result["ts"]})
        except Exception as exc:
            logger.exception("ingest failed", extra={"symbol": symbol})
            agent_status[symbol] = Status(symbol=symbol, state="failed", last_update=None, error_message=str(exc))
    asyncio.create_task(task())
    return IngestResponse(symbol=symbol, inserted=0, ts="queued")

@app.get("/status", response_model=list[Status])
async def status():
    return list(agent_status.values())

@app.get("/bars/{symbol}")
async def get_bars(symbol: str, limit: int = 100, offset: int = 0):
    if limit <= 0 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be 1..1000")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    df, total = db.bars_page(symbol, limit, offset)
    return {"rows": df.to_dict(orient="records"), "total": total, "limit": limit, "offset": offset}

@app.get("/symbols")
async def symbols():
    return db.list_symbols()

@app.get("/tables")
async def tables():
    return db.list_tables()

@app.get("/tables/{table}/rows")
async def table_rows(table: str, limit: int = 100, offset: int = 0):
    if limit <= 0 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be 1..1000")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    try:
        df, total = db.table_page(table, limit, offset)
    except ValueError:
        raise HTTPException(status_code=404, detail="table not found")
    return {"rows": df.to_dict(orient="records"), "total": total, "limit": limit, "offset": offset}

@app.get("/logs")
async def logs(limit: int = 100):
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive")
    with log_lock:
        return log_buffer[-limit:]

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
        <label for="lookback-years">lookback (years)</label><br/>
        <input id="start" placeholder="start ISO (optional)" style="width:18ch" />
        <input id="end" placeholder="end ISO (optional)" style="width:18ch" />
        <button onclick="ingest()">Start ingest</button>
        <span id="ingest-result"></span>
      </section>

      <section>
        <h3>Agent status</h3>
        <button onclick="refreshStatus()">Refresh</button>
        <ul id="status-list"></ul>
        <h4>Server logs</h4>
        <pre id="logs-box" style="background:#f7f7f7; padding:0.5rem; max-height:240px; overflow:auto;"></pre>
      </section>

      <section>
        <h3>Latest bars</h3>
        <input id="bars-symbol" placeholder="symbol" />
        <input id="bars-limit" type="number" value="20" min="1" max="1000" />
        <button onclick="loadBars(0)">Load</button>
        <button onclick="barsFirst()">First</button>
        <button onclick="barsPrev()">Prev</button>
        <button onclick="barsNext()">Next</button>
        <button onclick="barsLast()">Last</button>
        <span id="bars-page-info"></span>
        <table id="bars-table">
          <thead></thead>
          <tbody></tbody>
        </table>
      </section>

      <section>
        <h3>Tables</h3>
        <button onclick="loadTables()">Refresh tables</button>
        <ul id="tables-list"></ul>
        <div>
          <strong>Browse table</strong>
          <input id="table-name" placeholder="table name" />
          <input id="table-limit" type="number" value="20" min="1" max="1000" />
          <button onclick="loadTableRows(0)">Load</button>
          <button onclick="tableFirst()">First</button>
          <button onclick="tablePrev()">Prev</button>
          <button onclick="tableNext()">Next</button>
          <button onclick="tableLast()">Last</button>
          <span id="table-page-info"></span>
          <table id="table-rows">
            <thead></thead>
            <tbody></tbody>
          </table>
        </div>
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
          const startManualEl = document.getElementById("start");
          const endEl = document.getElementById("end");
          const startManual = startManualEl ? startManualEl.value.trim() : "";
          const end = endEl ? endEl.value.trim() : "";
          const lookback = parseFloat(document.getElementById("lookback-years").value);
          if (!symbol) { alert("Enter a symbol"); return; }
          const qs = new URLSearchParams();
          const startFromLookback = startManual || isoFromLookbackYears(lookback);
          if (startFromLookback) qs.append("start", startFromLookback);
          if (end) qs.append("end", end);
          try {
            console.log("ingest request", { symbol, start: startFromLookback, end });
            const res = await fetch(`/ingest/${symbol}?` + qs.toString(), { method: "POST" });
            const data = await res.json();
            document.getElementById("ingest-result").textContent = JSON.stringify(data);
          } catch (e) {
            console.error("ingest error", e);
            document.getElementById("ingest-result").textContent = "error: " + e;
          }
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
            if (s.error_message) {
              const err = document.createElement("div");
              err.style.color = "red";
              err.textContent = `error: ${s.error_message}`;
              li.appendChild(err);
            }
            list.appendChild(li);
          });
        }

        let barsState = {offset: 0, total: 0, limit: 20, symbol: ""};
        let tableState = {offset: 0, total: 0, limit: 20, table: ""};

        async function loadBars(offsetOverride) {
          const symbol = document.getElementById("bars-symbol").value.trim();
          const limit = parseInt(document.getElementById("bars-limit").value, 10);
          if (!symbol) { alert("Enter a symbol"); return; }
          const offset = offsetOverride ?? barsState.offset;
          const res = await fetch(`/bars/${symbol}?limit=${limit}&offset=${offset}`);
          const data = await res.json();
          barsState = {offset: data.offset, total: data.total, limit: data.limit, symbol};
          const table = document.getElementById("bars-table");
          table.querySelector("thead").innerHTML = "<tr><th>ts</th><th>open</th><th>high</th><th>low</th><th>close</th><th>volume</th></tr>";
          const tbody = table.querySelector("tbody");
          tbody.innerHTML = "";
          data.rows.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${r.ts}</td><td>${r.open}</td><td>${r.high}</td><td>${r.low}</td><td>${r.close}</td><td>${r.volume}</td>`;
            tbody.appendChild(tr);
          });
          document.getElementById("bars-page-info").textContent = `Showing ${offset + 1} to ${offset + data.rows.length} of ${data.total} rows`;
        }

        async function loadTables() {
          const res = await fetch("/tables");
          const data = await res.json();
          const list = document.getElementById("tables-list");
          list.innerHTML = "";
          data.forEach(t => {
            const li = document.createElement("li");
            li.textContent = t;
            list.appendChild(li);
          });
        }

        async function loadTableRows(offsetOverride) {
          const table = document.getElementById("table-name").value.trim();
          const limit = parseInt(document.getElementById("table-limit").value, 10);
          if (!table) { alert("Enter a table name"); return; }
          const offset = offsetOverride ?? tableState.offset;
          const res = await fetch(`/tables/${table}/rows?limit=${limit}&offset=${offset}`);
          const data = await res.json();
          tableState = {offset: data.offset, total: data.total, limit: data.limit, table};
          const tableEl = document.getElementById("table-rows");
          tableEl.querySelector("thead").innerHTML = "";
          const tbody = tableEl.querySelector("tbody");
          tbody.innerHTML = "";
          if (data.rows.length > 0) {
            Object.keys(data.rows[0]).forEach(key => {
              const th = document.createElement("th");
              th.textContent = key;
              tableEl.querySelector("thead").appendChild(th);
            });
            data.rows.forEach(r => {
              const tr = document.createElement("tr");
              Object.values(r).forEach(v => {
                const td = document.createElement("td");
                td.textContent = v;
                tr.appendChild(td);
              });
              tbody.appendChild(tr);
            });
            document.getElementById("table-page-info").textContent = `Showing ${offset + 1} to ${offset + data.rows.length} of ${data.total} rows`;
          } else {
            document.getElementById("table-page-info").textContent = "No rows found";
          }
        }

        function barsFirst() {
          loadBars(0);
        }

        function barsPrev() {
          loadBars(barsState.offset - barsState.limit);
        }

        function barsNext() {
          loadBars(barsState.offset + barsState.limit);
        }

        function barsLast() {
          loadBars(Math.floor((barsState.total - 1) / barsState.limit) * barsState.limit);
        }

        function tableFirst() {
          loadTableRows(0);
        }

        function tablePrev() {
          loadTableRows(tableState.offset - tableState.limit);
        }

        function tableNext() {
          loadTableRows(tableState.offset + tableState.limit);
        }

        function tableLast() {
          loadTableRows(Math.floor((tableState.total - 1) / tableState.limit) * tableState.limit);
        }

        setInterval(() => { refreshStatus(); refreshLogs(); }, 3000);
        refreshStatus();
        refreshLogs();
        loadTables();
      </script>
    </body>
    </html>
    """

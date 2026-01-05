from __future__ import annotations
import asyncio
import logging
import threading
import contextlib
import importlib.util
import shlex
import sys
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ..config import settings
from ..logging import configure_json_logger
from ..storage.duckdb_client import DuckDBClient
from ..ingestion.poller import IngestPoller
from ..ingestion.alpaca_client import get_alpaca_client

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
        if hasattr(record, "raw"):
            payload["raw"] = record.raw
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

class TestRunRequest(BaseModel):
    expression: str | None = None

    def targets(self) -> list[str]:
        if not self.expression:
            return []
        try:
            return shlex.split(self.expression)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid test expression: {exc}") from exc


TEST_MAX_OUTPUT = 20_000


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

# In-memory agent status
agent_status: dict[str, Status] = {}
test_state_lock = asyncio.Lock()
test_state: dict[str, object] = {
    "status": "idle",
    "targets": [],
    "started_at": None,
    "completed_at": None,
    "returncode": None,
    "stdout": "",
    "stderr": "",
}
current_test_task: asyncio.Task | None = None


def _truncate_output(text: str) -> str:
  if len(text) <= TEST_MAX_OUTPUT:
    return text
  return text[-TEST_MAX_OUTPUT:]


async def _run_tests_task(targets: list[str]) -> None:
  global current_test_task
  if importlib.util.find_spec("pytest") is None:
    message = "pytest is not installed. Install dev extras with 'pip install -e .[dev]' or 'poetry install --with dev'."
    async with test_state_lock:
      test_state.update(
        {
          "status": "error",
          "completed_at": _now_iso(),
          "returncode": None,
          "stdout": "",
          "stderr": message,
        }
      )
      current_test_task = None
    logger.warning("pytest missing for test run")
    return
  cmd = [sys.executable, "-m", "pytest", *targets]
  logger.info("test run started", extra={"cmd": cmd})
  status = "failed"
  returncode: int | None = None
  stdout_text = ""
  stderr_text = ""
  try:
    proc = await asyncio.create_subprocess_exec(
      *cmd,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    returncode = proc.returncode
    stdout_text = stdout_bytes.decode(errors="replace")
    stderr_text = stderr_bytes.decode(errors="replace")
    status = "passed" if returncode == 0 else "failed"
  except Exception as exc:  # pragma: no cover - defensive
    logger.exception("test run crashed")
    stderr_text = f"Test runner error: {exc}"
    status = "error"
  stdout_text = _truncate_output(stdout_text)
  stderr_text = _truncate_output(stderr_text)
  async with test_state_lock:
    test_state.update(
      {
        "status": status,
        "completed_at": _now_iso(),
        "returncode": returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
      }
    )
    current_test_task = None
  logger.info("test run finished", extra={"status": status, "returncode": returncode})


def _tests_snapshot() -> dict[str, object]:
  return {key: test_state.get(key) for key in test_state}


@app.get("/tests")
async def get_tests_state():
  async with test_state_lock:
    return _tests_snapshot()


@app.post("/tests/run")
async def trigger_tests(request: TestRunRequest):
  targets = request.targets()
  async with test_state_lock:
    global current_test_task
    if test_state["status"] == "running":
      raise HTTPException(status_code=409, detail="test run already in progress")
    test_state.update(
      {
        "status": "running",
        "targets": targets,
        "started_at": _now_iso(),
        "completed_at": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
      }
    )
    current_test_task = asyncio.create_task(_run_tests_task(targets))
    snapshot = _tests_snapshot()
  return snapshot

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

@app.delete("/bars/{symbol}")
async def delete_bars_symbol(symbol: str):
    deleted = db.delete_symbol(symbol)
    return {"symbol": symbol, "deleted": deleted}

@app.delete("/bars")
async def delete_all_bars():
    db.delete_all()
    return {"deleted": "all"}

@app.get("/debug/alpaca/raw")
async def get_alpaca_raw_debug():
    return {"alpaca_debug_raw": settings.alpaca_debug_raw}

@app.post("/debug/alpaca/raw")
async def set_alpaca_raw_debug(enabled: bool):
    settings.alpaca_debug_raw = enabled
    logger.info("alpaca raw debug toggled", extra={"enabled": enabled})
    return {"alpaca_debug_raw": settings.alpaca_debug_raw}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    base = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Local Stock DB Dashboard</title>
      <style>
        :root {{ color-scheme: light; }}
        body {{ font-family: "Inter", system-ui, sans-serif; margin: 0; background: #f5f7fb; color: #0f172a; }}
        header {{ padding: 1.25rem 1.5rem; background: #0f172a; color: #e2e8f0; }}
        h1 {{ margin: 0; font-size: 1.25rem; }}
        main {{ padding: 1rem 1.5rem; display: grid; gap: 1rem; }}
        section {{ background: #fff; border-radius: 10px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); padding: 1rem; }}
        .row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }}
        label {{ font-weight: 600; }}
        input, button {{ padding: 0.4rem 0.55rem; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 0.95rem; }}
        button {{ background: #2563eb; color: #fff; border: none; cursor: pointer; }}
        button:hover {{ background: #1d4ed8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 0.35rem 0.5rem; font-size: 0.9rem; }}
        .badge {{ padding: 0.15rem 0.5rem; border-radius: 999px; background: #e0f2fe; color: #075985; font-size: 0.8rem; display: inline-block; }}
        pre {{ white-space: pre-wrap; font-size: 0.85rem; background: #0b1224; color: #e2e8f0; padding: 0.75rem; border-radius: 8px; max-height: 260px; overflow: auto; }}
        #logs-container {{ max-height: 300px; overflow: auto; background: #0b1224; color: #e2e8f0; border-radius: 8px; padding: 0.4rem; }}
        ul {{ padding-left: 1.1rem; }}
      </style>
    </head>
    <body>
      <header>
        <h1>Local Stock Price Database</h1>
        <div class="row" style="gap:0.75rem; margin-top:0.35rem;">
          <span class="badge">Alpaca feed: __ALPACA_FEED__</span>
          <span class="badge">DuckDB: __DUCKDB_PATH__</span>
        </div>
      </header>
      <main>
        <section>
          <h3>Ingest</h3>
          <div class="row">
            <label>Symbol</label><input id="symbol" placeholder="e.g. AAPL" />
            <label>Lookback (years)</label><input id="lookback-years" type="number" value="1" step="0.25" min="0.1" style="width:8ch" />
            <label>Start (ISO)</label><input id="start" placeholder="optional" style="width:18ch" />
            <label>End (ISO)</label><input id="end" placeholder="optional" style="width:18ch" />
            <button onclick="ingest()">Start ingest</button>
            <span id="ingest-result"></span>
          </div>
          <div class="row" style="margin-top:0.5rem;">
            <button onclick="deleteSymbol()">Delete symbol data</button>
            <button onclick="deleteAll()">Delete ALL data</button>
          </div>
          <div class="row" style="margin-top:0.5rem;">
            <label for="alpaca-debug">Alpaca raw debug</label>
            <input type="checkbox" id="alpaca-debug" onchange="toggleAlpacaDebug(this.checked)" />
            <span id="alpaca-debug-state" class="badge"></span>
          </div>
        </section>

        <section>
          <h3>Agent status & Logs</h3>
          <div class="row">
            <button onclick="refreshStatus()">Refresh</button>
            <button onclick="toggleLogs()">Collapse/Expand logs</button>
          </div>
          <ul id="status-list"></ul>
           <h4>Server logs</h4>
          <div id="logs-container" style="display:block;">
            <div id="logs-box"></div>
          </div>
        </section>

        <section>
          <h3>Tests</h3>
          <div class="row">
            <input id="tests-expression" placeholder="pytest targets (optional)" style="flex:1; min-width:200px;" />
            <button onclick="runAllTests()">Run all tests</button>
            <button onclick="runSelectedTests()">Run selection</button>
            <span id="tests-status" class="badge"></span>
          </div>
          <div style="margin-top:0.5rem;">
            <div id="tests-meta" style="font-size:0.85rem; color:#475569;"></div>
            <details open style="margin-top:0.5rem;">
              <summary>stdout</summary>
              <pre id="tests-stdout"></pre>
            </details>
            <details style="margin-top:0.5rem;">
              <summary>stderr</summary>
              <pre id="tests-stderr"></pre>
            </details>
          </div>
        </section>

        <section>
          <h3>Latest bars</h3>
          <div class="row">
            <input id="bars-symbol" placeholder="symbol" />
            <input id="bars-limit" type="number" value="20" min="1" max="1000" />
            <button onclick="loadBars(0)">Load</button>
            <button onclick="barsFirst()">First</button>
            <button onclick="barsPrev()">Prev</button>
            <button onclick="barsNext()">Next</button>
            <button onclick="barsLast()">Last</button>
            <span id="bars-page-info"></span>
          </div>
          <table id="bars-table"><thead></thead><tbody></tbody></table>
        </section>

        <section>
          <h3>Tables</h3>
          <div class="row">
            <button onclick="loadTables()">Refresh tables</button>
            <ul id="tables-list"></ul>
          </div>
          <div class="row" style="margin-top:0.5rem;">
            <input id="table-name" placeholder="table name" />
            <input id="table-limit" type="number" value="20" min="1" max="1000" />
            <button onclick="loadTableRows(0)">Load</button>
            <button onclick="tableFirst()">First</button>
            <button onclick="tablePrev()">Prev</button>
            <button onclick="tableNext()">Next</button>
            <button onclick="tableLast()">Last</button>
            <span id="table-page-info"></span>
          </div>
          <table id="table-rows"><thead></thead><tbody></tbody></table>
        </section>
      </main>

      <script>
        function isoFromLookbackYears(years) { if (!years || isNaN(years) || years <= 0) return null; const d = new Date(); d.setFullYear(d.getFullYear() - years); return d.toISOString(); }

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

        async function deleteSymbol() {
          const symbol = document.getElementById("symbol").value.trim();
          if (!symbol) { alert("Enter a symbol"); return; }
          if (!confirm(`Delete data for ${symbol}?`)) return;
          await fetch(`/bars/${symbol}`, { method: "DELETE" });
          refreshStatus(); loadTables();
        }

        async function deleteAll() {
          if (!confirm("Delete ALL data?")) return;
          await fetch(`/bars`, { method: "DELETE" });
          refreshStatus(); loadTables();
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
          const head = document.querySelector("#bars-table thead");
          const body = document.querySelector("#bars-table tbody");
          head.innerHTML = ""; body.innerHTML = "";
          if (!data.rows || data.rows.length === 0) { body.innerHTML = "<tr><td>No data</td></tr>"; document.getElementById("bars-page-info").textContent = ""; return; }
          const cols = Object.keys(data.rows[0]);
          const trHead = document.createElement("tr"); cols.forEach(c => { const th = document.createElement("th"); th.textContent = c; trHead.appendChild(th); }); head.appendChild(trHead);
          data.rows.forEach(row => { const tr = document.createElement("tr"); cols.forEach(c => { const td = document.createElement("td"); td.textContent = row[c]; tr.appendChild(td); }); body.appendChild(tr); });
          const end = Math.min(barsState.offset + barsState.limit, barsState.total);
          document.getElementById("bars-page-info").textContent = `Rows ${barsState.offset + 1}-${end} of ${barsState.total}`;
        }
        function barsFirst() { loadBars(0); }
        function barsPrev() { loadBars(Math.max(0, barsState.offset - barsState.limit)); }
        function barsNext() { const o = barsState.offset + barsState.limit; if (o < barsState.total) loadBars(o); }
        function barsLast() { loadBars(Math.max(0, barsState.total - barsState.limit)); }

        async function loadTables() {
          const res = await fetch("/tables");
          const data = await res.json();
          const list = document.getElementById("tables-list");
          list.innerHTML = "";
          data.forEach(t => {
            const li = document.createElement("li");
            const btn = document.createElement("button");
            btn.textContent = t;
            btn.onclick = () => { document.getElementById("table-name").value = t; loadTableRows(0); };
            li.appendChild(btn);
            list.appendChild(li);
          });
        }

        async function loadTableRows(offsetOverride) {
          const table = document.getElementById("table-name").value.trim();
          const limit = parseInt(document.getElementById("table-limit").value, 10);
          if (!table) { alert("Enter a table name"); return; }
          const offset = offsetOverride ?? tableState.offset;
          const res = await fetch(`/tables/${table}/rows?limit=${limit}&offset=${offset}`);
          if (res.status === 404) { alert("Table not found"); return; }
          const data = await res.json();
          tableState = {offset: data.offset, total: data.total, limit: data.limit, table};
          const head = document.querySelector("#table-rows thead");
          const body = document.querySelector("#table-rows tbody");
          head.innerHTML = ""; body.innerHTML = "";
          if (!data.rows || data.rows.length === 0) { body.innerHTML = "<tr><td>No data</td></tr>"; document.getElementById("table-page-info").textContent = ""; return; }
          const cols = Object.keys(data.rows[0]);
          const trHead = document.createElement("tr"); cols.forEach(c => { const th = document.createElement("th"); th.textContent = c; trHead.appendChild(th); }); head.appendChild(trHead);
          data.rows.forEach(row => { const tr = document.createElement("tr"); cols.forEach(c => { const td = document.createElement("td"); td.textContent = row[c]; tr.appendChild(td); }); body.appendChild(tr); });
          const end = Math.min(tableState.offset + tableState.limit, tableState.total);
          document.getElementById("table-page-info").textContent = `Rows ${tableState.offset + 1}-${end} of ${tableState.total}`;
        }
        function tableFirst() { loadTableRows(0); }
        function tablePrev() { loadTableRows(Math.max(0, tableState.offset - tableState.limit)); }
        function tableNext() { const o = tableState.offset + tableState.limit; if (o < tableState.total) loadTableRows(o); }
        function tableLast() { loadTableRows(Math.max(0, tableState.total - tableState.limit)); }

        async function refreshStatus() {
          const res = await fetch("/status");
          const data = await res.json();
          const list = document.getElementById("status-list");
          list.innerHTML = "";
          data.forEach(s => {
            const li = document.createElement("li");
            li.textContent = `${s.symbol}: ${s.state}` + (s.last_update ? ` @ ${s.last_update}` : "");
            if (s.error_message) { const err = document.createElement("div"); err.style.color = "red"; err.textContent = `error: ${s.error_message}`; li.appendChild(err); }
            list.appendChild(li);
          });
        }

        async function refreshLogs() {
          const res = await fetch("/logs?limit=200");
          const data = await res.json();
          const box = document.getElementById("logs-box");
          box.innerHTML = "";
          [...data].reverse().forEach((l, idx) => {
             const row = document.createElement("div");
             row.style.marginBottom = "0.35rem";
             const summary = document.createElement("div");
             summary.textContent = `[${l.level}] ${l.message}`;
             row.appendChild(summary);
             if (l.raw) {
               const btn = document.createElement("button");
               btn.textContent = "show raw";
               btn.style.marginTop = "0.2rem";
               const pre = document.createElement("pre");
               pre.style.display = "none";
               try { pre.textContent = JSON.stringify(JSON.parse(l.raw), null, 2); }
               catch { pre.textContent = l.raw; }
               btn.onclick = () => { pre.style.display = pre.style.display === "none" ? "block" : "none"; };
               row.appendChild(btn);
               row.appendChild(pre);
             }
             box.appendChild(row);
           });
           box.scrollTop = box.scrollHeight;
        }

        function toggleLogs() {
          const container = document.getElementById("logs-container");
          container.style.display = container.style.display === "none" ? "block" : "none";
        }

        function renderTests(state) {
          const statusBadge = document.getElementById("tests-status");
          statusBadge.textContent = state.status || "unknown";
          statusBadge.style.background = state.status === "passed" ? "#dcfce7" : state.status === "failed" || state.status === "error" ? "#fee2e2" : "#e0f2fe";
          statusBadge.style.color = state.status === "passed" ? "#166534" : state.status === "failed" || state.status === "error" ? "#b91c1c" : "#075985";
          const meta = document.getElementById("tests-meta");
          const targets = state.targets && state.targets.length ? state.targets.join(" ") : "all";
          const started = state.started_at ? `started: ${state.started_at}` : "";
          const completed = state.completed_at ? `finished: ${state.completed_at}` : "";
          const rc = state.returncode !== null && state.returncode !== undefined ? `exit code: ${state.returncode}` : "";
          meta.textContent = [targets ? `targets: ${targets}` : "", started, completed, rc].filter(Boolean).join(" • ");
          document.getElementById("tests-stdout").textContent = state.stdout || "";
          document.getElementById("tests-stderr").textContent = state.stderr || "";
        }

        async function refreshTests() {
          try {
            const res = await fetch("/tests");
            if (!res.ok) return;
            const data = await res.json();
            renderTests(data);
          } catch (err) {
            console.error("failed to refresh tests", err);
          }
        }

        async function runTests(expression) {
          const payload = expression ? { expression } : {};
          try {
            const res = await fetch("/tests/run", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
            if (!res.ok) {
              const detail = await res.json().catch(() => ({}));
              alert(detail.detail || "Failed to start tests");
              return;
            }
            const data = await res.json();
            renderTests(data);
          } catch (err) {
            alert("Unable to start tests");
          }
        }

        function runAllTests() {
          runTests(null);
        }

        function runSelectedTests() {
          const expr = document.getElementById("tests-expression").value.trim();
          if (!expr) {
            alert("Enter pytest targets to run selection");
            return;
          }
          runTests(expr);
        }

        async function loadDebugToggle() {
          const res = await fetch("/debug/alpaca/raw");
          const data = await res.json();
          const cb = document.getElementById("alpaca-debug");
          const badge = document.getElementById("alpaca-debug-state");
          cb.checked = !!data.alpaca_debug_raw;
          badge.textContent = `alpaca_debug_raw=${data.alpaca_debug_raw}`;
        }

        async function toggleAlpacaDebug(enabled) {
          await fetch(`/debug/alpaca/raw?enabled=${enabled}`, { method: "POST" });
          loadDebugToggle();
        }

        setInterval(() => { refreshStatus(); refreshLogs(); refreshTests(); }, 3000);
        refreshStatus(); refreshLogs(); refreshTests(); loadTables(); loadDebugToggle();
      </script>
    </body>
    </html>
    """
    html = base.replace("__ALPACA_FEED__", str(settings.alpaca_feed)).replace("__DUCKDB_PATH__", str(settings.duckdb_path))
    return HTMLResponse(html)

live_task: asyncio.Task | None = None

async def live_updater():
    if not (settings.alpaca_key_id and settings.alpaca_secret_key):
        logger.info("live updater disabled; missing alpaca credentials")
        return
    client = get_alpaca_client()
    try:
        while True:
            try:
                clock = await client.get_clock()
                is_open = clock.get("is_open", False)
                next_open = clock.get("next_open")
                next_close = clock.get("next_close")
                if not is_open:
                    sleep_for = 60
                    if next_open:
                        try:
                            ts = datetime.fromisoformat(next_open.replace("Z", "+00:00"))
                            delta = (ts - datetime.now(timezone.utc)).total_seconds()
                            sleep_for = max(30, min(delta, 600)) if delta > 0 else 60
                        except Exception:
                            sleep_for = 60
                    logger.info("market closed; sleeping", extra={"sleep_seconds": sleep_for, "next_open": next_open})
                    await asyncio.sleep(sleep_for)
                    continue
                symbols = db.list_symbols()
                if symbols:
                    result = await poller.run_live_batch(symbols)
                    logger.info("live batch complete", extra={"symbols": result["symbols"], "inserted": result["inserted"]})
                else:
                    logger.info("live batch skipped; no symbols")
                if next_close:
                    try:
                        ts_close = datetime.fromisoformat(next_close.replace("Z", "+00:00"))
                        if ts_close < datetime.now(timezone.utc):
                            await asyncio.sleep(60)
                            continue
                    except Exception:
                        pass
                await asyncio.sleep(60)
            except Exception:
                logger.exception("live updater loop error")
                await asyncio.sleep(60)
    finally:
        await client.aclose()

@app.on_event("startup")
async def _startup():
    global live_task
    live_task = asyncio.create_task(live_updater())

@app.on_event("shutdown")
async def _shutdown():
    global live_task
    if live_task:
        live_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await live_task
    live_task = None

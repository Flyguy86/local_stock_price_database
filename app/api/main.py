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
from pydantic import BaseModel, Field
from ..config import settings
from ..logging import configure_json_logger
from ..storage.duckdb_client import DuckDBClient
from ..storage.backfill import BackfillManager
from ..ingestion.poller import IngestPoller
from ..ingestion.alpaca_client import get_alpaca_client

logger = configure_json_logger(settings.log_level)
LOG_BUFFER_MAX = 200
log_buffer: list[dict] = []
log_lock = threading.Lock()

class InMemoryHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "GET /logs" in msg or "GET /status" in msg:
            return

        # Resolve timestamp
        ts = getattr(record, "ts", None)
        if not ts:
            dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
            ts = dt.isoformat()
            
        payload = {
            "ts": ts,
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

# Attach to both 'app' (application logs) and 'uvicorn' (requests)
logger.addHandler(InMemoryHandler())
logging.getLogger("uvicorn").addHandler(InMemoryHandler())
logging.getLogger("uvicorn.access").addHandler(InMemoryHandler())
logging.getLogger("uvicorn.error").addHandler(InMemoryHandler())
db = DuckDBClient(settings.duckdb_path, settings.parquet_dir)
poller = IngestPoller(db)
backfill_manager = BackfillManager(settings.duckdb_path, settings.parquet_dir)
app = FastAPI(title="local_stock_price_database")

class IngestResponse(BaseModel):
    symbol: str
    inserted: int
    ts: str

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

class Status(BaseModel):
    symbol: str
    state: str
    description: str | None = None
    last_update: str | None = None
    error_message: str | None = None
    updated_at: str = Field(default_factory=_now_iso)

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


# In-memory agent status
agent_status: dict[str, Status] = {}
running_tasks: dict[str, asyncio.Task] = {}  # symbol -> task
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
    
    if symbol in running_tasks and not running_tasks[symbol].done():
        raise HTTPException(status_code=409, detail=f"Task for {symbol} is already running")

    logger.info("ingest request queued", extra={"symbol": symbol, "start": start, "end": end})
    agent_status[symbol] = Status(symbol=symbol, state="queued", description="Initializing...", last_update=None)
    
    async def task():
        agent_status[symbol] = Status(symbol=symbol, state="running", description=f"Backfilling {start or 'max'} -> {end or 'now'}", last_update=None)
        logger.info("ingest task started", extra={"symbol": symbol, "start": start, "end": end})
        try:
            # 1. Run price history backfill
            result = await poller.run_history(symbol, start=start, end=end)
            
            # 2. Run earnings fetch (best effort)
            agent_status[symbol] = Status(symbol=symbol, state="running", description="Fetching earnings", last_update=None)
            try:
                await poller.run_earnings(symbol)
            except Exception as e:
                logger.warning("earnings fetch failed during ingest", extra={"symbol": symbol, "error": str(e)})

            agent_status[symbol] = Status(symbol=symbol, state="succeeded", description="Complete", last_update=result["ts"])
            logger.info("ingest task succeeded", extra={"symbol": symbol, "inserted": result["inserted"], "ts": result["ts"]})
        except asyncio.CancelledError:
             logger.warning("ingest task cancelled", extra={"symbol": symbol})
             agent_status[symbol] = Status(symbol=symbol, state="stopped", description="User cancelled", last_update=None)
        except Exception as exc:
            logger.exception("ingest failed", extra={"symbol": symbol})
            agent_status[symbol] = Status(symbol=symbol, state="failed", description="Failed", last_update=None, error_message=str(exc))
        finally:
             if symbol in running_tasks:
                 del running_tasks[symbol]

    t = asyncio.create_task(task())
    running_tasks[symbol] = t
    return IngestResponse(symbol=symbol, inserted=0, ts="queued")

@app.post("/stop/{symbol}")
async def stop_task(symbol: str):
    if symbol in running_tasks and not running_tasks[symbol].done():
         running_tasks[symbol].cancel()
         return {"status": "stopping", "symbol": symbol}
    return {"status": "no_running_task", "symbol": symbol}


@app.post("/ingest/earnings/{symbol}")
async def ingest_earnings_endpoint(symbol: str):
    if symbol in running_tasks and not running_tasks[symbol].done():
        raise HTTPException(status_code=409, detail=f"Task for {symbol} is already running")

    logger.info("ingest earnings queued", extra={"symbol": symbol})
    agent_status[symbol] = Status(symbol=symbol, state="queued", description="Queued earnings fetch", last_update=None)
    
    async def task():
        agent_status[symbol] = Status(symbol=symbol, state="running", description="Fetching earnings", last_update=None)
        try:
            result = await poller.run_earnings(symbol)
            agent_status[symbol] = Status(symbol=symbol, state="succeeded", description="Earnings complete", last_update=result["ts"])
        except asyncio.CancelledError:
             agent_status[symbol] = Status(symbol=symbol, state="stopped", description="User cancelled", last_update=None)
        except Exception as exc:
            logger.exception("ingest earnings failed", extra={"symbol": symbol})
            agent_status[symbol] = Status(symbol=symbol, state="failed", description="Earnings failed", last_update=None, error_message=str(exc))
        finally:
             if symbol in running_tasks:
                 del running_tasks[symbol]
    
    t = asyncio.create_task(task())
    running_tasks[symbol] = t
    return {"status": "queued", "symbol": symbol}

@app.post("/backfill/{symbol}")
async def backfill_missing_data(symbol: str, max_iterations: int = 100):
    """
    Backfill missing 1-minute bars for a symbol during market hours.
    Fills gaps with the mean of adjacent bars.
    """
    if symbol in running_tasks and not running_tasks[symbol].done():
        raise HTTPException(status_code=409, detail=f"Task for {symbol} is already running")
    
    if max_iterations <= 0 or max_iterations > 1000:
        raise HTTPException(status_code=400, detail="max_iterations must be 1..1000")

    logger.info("backfill request queued", extra={"symbol": symbol, "max_iterations": max_iterations})
    agent_status[symbol] = Status(symbol=symbol, state="queued", description="Queued backfill", last_update=None)
    
    async def task():
        agent_status[symbol] = Status(symbol=symbol, state="running", description="Scanning for missing bars", last_update=None)
        try:
            # Run backfill in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                backfill_manager.backfill_symbol,
                symbol,
                max_iterations
            )
            
            agent_status[symbol] = Status(
                symbol=symbol,
                state="succeeded",
                description=f"Backfilled {result['filled']} bars",
                last_update=_now_iso()
            )
            logger.info("backfill task succeeded", extra={
                "symbol": symbol,
                "filled": result["filled"],
                "iterations": result["iterations"]
            })
        except asyncio.CancelledError:
            logger.warning("backfill task cancelled", extra={"symbol": symbol})
            agent_status[symbol] = Status(symbol=symbol, state="stopped", description="User cancelled", last_update=None)
        except Exception as exc:
            logger.exception("backfill failed", extra={"symbol": symbol})
            agent_status[symbol] = Status(
                symbol=symbol,
                state="failed",
                description="Backfill failed",
                last_update=None,
                error_message=str(exc)
            )
        finally:
            if symbol in running_tasks:
                del running_tasks[symbol]
    
    t = asyncio.create_task(task())
    running_tasks[symbol] = t
    return {"status": "queued", "symbol": symbol, "max_iterations": max_iterations}

@app.get("/status", response_model=list[Status])
async def status():
    now = datetime.now(timezone.utc)
    to_remove = []
    for sym, s in agent_status.items():
        if s.state in ("succeeded", "failed", "stopped"):
            try:
                ts = datetime.fromisoformat(s.updated_at)
                if (now - ts).total_seconds() > 1800:  # 30 minutes
                    to_remove.append(sym)
            except Exception:
                pass
    for sym in to_remove:
        del agent_status[sym]
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
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        :root {
          --bg: #0f172a;
          --bg-card: #1e293b;
          --text: #e2e8f0;
          --text-muted: #94a3b8;
          --border: #334155;
          --primary: #3b82f6;
          --primary-hover: #2563eb;
          --danger: #ef4444;
          --success: #10b981;
        }
        body { margin: 0; font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }
        * { box-sizing: border-box; }
        
        /* Layout */
        .layout { display: grid; grid-template-rows: auto 1fr; height: 100vh; }
        header { border-bottom: 1px solid var(--border); padding: 1rem 1.5rem; background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(8px); display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 10; }
        
        h1 { margin: 0; font-size: 1.25rem; font-weight: 600; background: linear-gradient(to right, #60a5fa, #e879f9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        main { padding: 1.5rem; overflow-y: auto; display: grid; gap: 1.5rem; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); align-content: start; }
        .full-width { grid-column: 1 / -1; }
        
        section { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; display: flex; flex-direction: column; gap: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        
        h2 { margin: 0; font-size: 1rem; font-weight: 600; color: var(--text); display: flex; align-items: center; gap: 0.5rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); }
        h4 { margin: 0 0 0.5rem 0; font-size:0.9rem; color:#cbd5e1; }

        /* Controls */
        .row { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }
        .group { display: flex; align-items: center; gap: 0.5rem; background: rgba(0,0,0,0.2); padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid var(--border); }
        
        label { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; }
        input, select { background: transparent; border: none; color: var(--text); font-family: inherit; font-size: 0.9rem; padding: 0.25rem; outline: none; }
        input:focus { color: #fff; }
        input[type="number"] { width: 60px; }
        input[placeholder] { min-width: 100px; }
        
        button { background: var(--primary); color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 500; cursor: pointer; transition: all 0.2s; font-size: 0.85rem; }
        button:hover { background: var(--primary-hover); transform: translateY(-1px); }
        button.secondary { background: var(--border); color: var(--text); }
        button.secondary:hover { background: #475569; }
        button.danger { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
        button.danger:hover { background: rgba(239, 68, 68, 0.3); }
        button.sm { padding: 0.25rem 0.5rem; font-size: 0.8rem; }
        
        /* Data Tables */
        .table-container { overflow-x: auto; border: 1px solid var(--border); border-radius: 6px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th, td { text-align: left; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
        th { background: rgba(0,0,0,0.2); color: var(--text-muted); font-weight: 600; position: sticky; top: 0; }
        tr:hover { background: rgba(255,255,255,0.02); }
        
        /* Utils */
        .badge { display: inline-flex; align-items: center; padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500; background: var(--border); color: var(--text-muted); user-select: none; }
        .badge.green { background: rgba(16, 185, 129, 0.2); text: #6ee7b7; color: #34d399; }
        .badge.red { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
        .badge.blue { background: rgba(59, 130, 246, 0.2); color: #93c5fd; }
        
        pre { background: rgba(0,0,0,0.3); padding: 0.75rem; border-radius: 6px; overflow: auto; margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; border: 1px solid var(--border); }
        
        /* Logs */
        #logs-box { height: 200px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; display: flex; flex-direction: column; gap: 0.25rem; background: #0f172a; padding: 0.5rem; border-radius: 4px; }
        .log-entry { padding: 0.25rem 0.5rem; border-radius: 4px; }
        .log-entry:hover { background: rgba(255,255,255,0.05); }
        .log-info { color: #94a3b8; }
        .log-warning { color: #fcd34d; }
        .log-error { color: #fca5a5; }
        
        /* Status List */
        .status-grid { display: flex; flex-direction: column; gap: 0.5rem; max-height: 300px; overflow-y: auto; padding-right: 0.25rem; }
        .status-card { background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: 6px; border: 1px solid var(--border); display: flex; justify-content: space-between; align-items:center; }
        .status-card .left { display: flex; flex-direction: column; gap: 0.25rem; }
        .status-card .sym { font-weight: 700; font-size: 1rem; color: #fff; }
        .status-card .desc { font-size: 0.8rem; color: var(--text-muted); }
        .status-card .right { display: flex; align-items:center; gap: 0.5rem; }
        
        /* Tabs */
        .tabs { display: flex; gap: 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem; }
        .tab { padding: 0.5rem 0; cursor: pointer; color: var(--text-muted); border-bottom: 2px solid transparent; transition: all 0.2s; }
        .tab:hover { color: var(--text); }
        .tab.active { color: var(--primary); border-bottom-color: var(--primary); }
        
        /* Split view specific */
        .split { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        @media (max-width: 900px) { .split { grid-template-columns: 1fr; } }
      </style>
    </head>
    <body>
      <div class="layout">
        <header>
            <div style="display:flex; align-items:center; gap:1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                <h1>Local Stock DB</h1>
            </div>
            <div class="row">
                <div class="badge blue">Feed: <span>__ALPACA_FEED__</span></div>
                <div class="badge blue">DB: <span>__DUCKDB_PATH__</span></div>
                <a href="/docs" target="_blank" class="badge" style="text-decoration:none; cursor:pointer;">API Docs &#8599;</a>
            </div>
        </header>
        
        <main>
            <!-- Ingest Control & Status Split -->
            <div class="full-width split">
                <!-- Control -->
                <section>
                    <h2>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        Ingest Control
                    </h2>
                    <div style="display:flex; flex-direction:column; gap:1rem;">
                        <div class="row">
                            <div class="group" style="flex: 1;">
                                <label>Symbol</label>
                                <input id="ingest-symbol" placeholder="e.g. NVDA" style="width: 80px; text-transform:uppercase;">
                            </div>
                        </div>
                        
                        <div class="row">
                             <div class="group">
                                <label>Lookback (Yrs)</label>
                                <input id="lookback-years" type="number" value="10" step="0.5">
                            </div>
                            <div class="group" style="flex: 1;">
                                 <label>Dates</label>
                                 <input id="start" placeholder="YYYY-MM-DD">
                                 <span>to</span>
                                 <input id="end" placeholder="YYYY-MM-DD">
                            </div>
                        </div>
                        
                        <div class="row">
                            <button id="ingest-button" onclick="ingest()" style="flex:1">Ingest Bars</button>
                            <button onclick="ingestEarnings()" class="secondary" title="Update Earnings Only">$ Earnings</button>
                        </div>
                        <div id="ingest-result" style="font-size: 0.8rem; color: var(--success); min-height: 1.2em;"></div>
                    </div>
                </section>

                <!-- Processes & Logs -->
                <section style="display:flex; flex-direction:column; min-height:400px;">
                    <h2 style="margin-bottom:0;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                        System Activity
                        <div style="margin-left:auto; display:flex; gap:0.5rem; align-items:center;">
                            <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.8rem; cursor:pointer;">
                                <input type="checkbox" id="alpaca-debug" onchange="toggleAlpacaDebug(this.checked)"> Raw Debug
                            </label>
                        </div>
                    </h2>
                    
                    <div style="flex:1; display:flex; flex-direction:column; gap:1rem;">
                        <div>
                             <h4>Running Processes</h4>
                             <div id="status-grid" class="status-grid">
                                <div style="color:var(--text-muted); font-size:0.8rem; text-align:center; padding:1rem;">No active processes</div>
                             </div>
                        </div>
                        
                        <div style="flex:1; display:flex; flex-direction:column; min-height:200px;">
                             <h4>Live Logs</h4>
                             <div id="logs-box" style="flex:1;"></div>
                        </div>
                    </div>
                </section>
            </div>

            <!-- Data Viewer -->
            <section class="full-width" style="min-height: 400px;">
                <div class="row" style="border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 0;">
                    <div class="tabs" style="margin: 0; border: none;">
                        <div class="tab active" onclick="setMode('bars', this)">Price Bars</div>
                        <div class="tab" onclick="setMode('tables', this)">Tables</div>
                    </div>
                    <div style="margin-left: auto;" class="row">
                         <div class="group" id="bars-controls">
                            <label>Sym</label>
                            <input id="bars-symbol" placeholder="AAPL" value="AAPL" onchange="loadData()">
                         </div>
                         <div class="group" id="tables-controls" style="display:none;">
                            <select id="tables-select" onchange="loadData()"></select>
                         </div>
                         <div class="group">
                            <label>Limit</label>
                            <input id="limit" type="number" value="20" onchange="loadData()">
                         </div>
                    </div>
                </div>
                
                <div class="table-container" style="flex: 1;">
                    <table id="data-table">
                        <thead></thead>
                        <tbody></tbody>
                    </table>
                </div>
                
                <div class="row" style="justify-content: space-between; padding-top: 0.5rem;">
                    <span id="page-info" class="badge"></span>
                    <div class="row" style="gap: 0.25rem;">
                        <button class="sm secondary" onclick="nav('first')">&laquo;</button>
                        <button class="sm secondary" onclick="nav('prev')">&lsaquo;</button>
                        <button class="sm secondary" onclick="nav('next')">&rsaquo;</button>
                        <button class="sm secondary" onclick="nav('last')">&raquo;</button>
                    </div>
                </div>
            </section>
            
            <!-- Tests Section (Mini) -->
            <section class="full-width">
                <h2>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    Tests
                    <span id="tests-status" class="badge" style="margin-left:1rem;"></span>
                </h2>
                <div class="row">
                    <input id="tests-expression" placeholder="pytest matching..." style="flex: 1; background: rgba(0,0,0,0.2); border: 1px solid var(--border); padding: 0.5rem; border-radius: 4px; color: var(--text);">
                    <button class="sm" onclick="runAllTests()">Run All</button>
                    <button class="sm secondary" onclick="runSelectedTests()">Run Selected</button>
                </div>
                 <details>
                    <summary style="cursor: pointer; font-size: 0.8rem; color: var(--text-muted); margin-top:0.5rem;">Output</summary>
                    <pre id="tests-stdout" style="margin-top: 0.5rem; max-height: 200px; color: #cbd5e1;"></pre>
                    <pre id="tests-stderr" style="margin-top: 0.5rem; max-height: 200px; color: #fca5a5;"></pre>
                </details>
            </section>
            
            <!-- Danger Zone -->
            <section class="full-width" style="border-color: rgba(239, 68, 68, 0.3);">
                <h2 style="color: #fca5a5; border-color: rgba(239, 68, 68, 0.3);">Danger Zone</h2>
                <div class="row">
                    <button class="danger" onclick="deleteSymbol()">Delete Symbol Data</button>
                    <button class="danger" onclick="deleteAll()">Reset Entire Database</button>
                </div>
            </section>
        </main>
      </div>

      <script>
        // State
        const state = { mode: 'bars', offset: 0, limit: 20, total: 0 };
        
        // Utils
        const $ = id => document.getElementById(id);
        const isoFromYears = y => { const d = new Date(); d.setFullYear(d.getFullYear() - y); return d.toISOString(); };

        // Initialization
        window.onload = () => {
             refreshStatus();
             refreshLogs();
             loadTablesList();
             loadData();
             loadDebugToggle();
             // Auto-refresh loops
             setInterval(refreshStatus, 2000); 
             setInterval(refreshLogs, 2000);
        };

        // Actions
        async function ingest() {
            const sym = $('ingest-symbol').value.trim().toUpperCase();
            if(!sym) return alert('Symbol required');
            
            const start = $('start').value || isoFromYears($('lookback-years').value);
            const end = $('end').value;
            
            $('ingest-button').disabled = true;
            $('ingest-result').innerText = 'Queueing...';
            
            try {
                let url = `/ingest/${sym}?start=${start}`;
                if(end) url += `&end=${end}`;
                const res = await fetch(url, {method: 'POST'});
                const data = await res.json();
                $('ingest-result').innerText = 'Queued: ' + data.ts;
                refreshStatus();
            } catch(e) {
                alert('Ingest failed: ' + e);
            } finally {
               $('ingest-button').disabled = false;
            }
        }

        async function stopTask(sym) {
            if(!confirm('Stop task for ' + sym + '?')) return;
            await fetch('/stop/' + sym, { method: 'POST' });
            refreshStatus();
        }
        
        async function ingestEarnings() {
             const sym = $('ingest-symbol').value.trim().toUpperCase();
             if(!sym) return alert('Symbol required');
             await fetch(`/ingest/earnings/${sym}`, { method: 'POST' });
             refreshStatus();
        }

        async function deleteSymbol() {
            const sym = $('ingest-symbol').value.trim().toUpperCase();
            if(!sym || !confirm(`Delete ${sym}?`)) return;
            await fetch(`/bars/${sym}`, {method: 'DELETE'});
            refreshStatus(); loadData();
        }

        async function deleteAll() {
            if(!confirm('NUKE THE DATABASE?')) return;
            await fetch('/bars', {method: 'DELETE'});
            refreshStatus(); loadData();
        }
        
        // Data Viewing
        function setMode(m, el) {
            state.mode = m;
            state.offset = 0;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            el.classList.add('active');
            
            $('bars-controls').style.display = m === 'bars' ? 'flex' : 'none';
            $('tables-controls').style.display = m === 'tables' ? 'flex' : 'none';
            loadData();
        }
        
        async function loadTablesList() {
            const res = await fetch('/tables');
            const tables = await res.json();
            const sel = $('tables-select');
            sel.innerHTML = tables.map(t => `<option value="${t}">${t}</option>`).join('');
        }
        
        async function loadData() {
            const limit = parseInt($('limit').value) || 20;
            let url = '';
            
            if(state.mode === 'bars') {
                const sym = $('bars-symbol').value.trim();
                url = `/bars/${sym}?limit=${limit}&offset=${state.offset}`;
            } else {
                const tbl = $('tables-select').value;
                if(!tbl) return;
                url = `/tables/${tbl}/rows?limit=${limit}&offset=${state.offset}`;
            }
            
            try {
                const res = await fetch(url);
                if(!res.ok) throw new Error('Fetch failed');
                const data = await res.json();
                
                state.total = data.total;
                state.limit = data.limit;
                state.offset = data.offset;
                
                renderTable(data.rows);
                
                const end = Math.min(state.offset + state.limit, state.total);
                $('page-info').innerText = `${state.offset + 1}-${end} of ${state.total}`;
            } catch(e) {
                // Ignore error on load if table doesn't exist yet
                $('data-table').querySelector('tbody').innerHTML = `<tr><td colspan="100" style="text-align:center">No data or error</td></tr>`;
            }
        }
        
        function renderTable(rows) {
            const thead = $('data-table').querySelector('thead');
            const tbody = $('data-table').querySelector('tbody');
            thead.innerHTML = '';
            tbody.innerHTML = '';
            
            if(!rows || !rows.length) {
                tbody.innerHTML = `<tr><td colspan="100" style="text-align:center; color: var(--text-muted)">No data</td></tr>`;
                return;
            }
            
            const cols = Object.keys(rows[0]);
            thead.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
            
            rows.forEach(r => {
                const tr = document.createElement('tr');
                tr.innerHTML = cols.map(c => `<td>${r[c]}</td>`).join('');
                tbody.appendChild(tr);
            });
        }
        
        function nav(dir) {
             if(dir === 'first') state.offset = 0;
             else if(dir === 'prev') state.offset = Math.max(0, state.offset - state.limit);
             else if(dir === 'next') { if(state.offset + state.limit < state.total) state.offset += state.limit; }
             else if(dir === 'last') state.offset = Math.max(0, state.total - state.limit);
             loadData();
        }
        
        // Tests
        async function runTests(expression) {
          const payload = expression ? { expression } : {};
          try {
            $('tests-status').innerText = "Running...";
            $('tests-status').className = "badge";
            const res = await fetch("/tests/run", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
            if (!res.ok) {
               const err = await res.json().catch(() => ({}));
               alert(err.detail || "Failed");
               return;
            }
            const data = await res.json();
            renderTests(data);
          } catch (err) { alert("Test start failed"); }
        }
        function runAllTests() { runTests(null); }
        function runSelectedTests() {
          const expr = $('tests-expression').value.trim();
          if (!expr) return alert("Enter target");
          runTests(expr);
        }
        async function refreshTests() {
            try {
                const res = await fetch("/tests");
                if(res.ok) renderTests(await res.json());
            } catch(e) {}
        }
        function renderTests(st) {
            const status = $('tests-status');
            status.innerText = st.status || "unknown";
            status.className = `badge ${st.status === "passed" ? "green" : st.status === "failed" ? "red" : ""}`;
            $('tests-stdout').innerText = st.stdout || "";
            $('tests-stderr').innerText = st.stderr || "";
        }
        
        // System Status
        async function refreshStatus() {
            const res = await fetch('/status');
            const data = await res.json();
            const grid = $('status-grid');
            
            // Only show active or recently failed/succeeded (last 5 mins) to keep list clean?
            // For now show all tracked
            
            if (data.length === 0) {
                 grid.innerHTML = '<div style="color:var(--text-muted); font-size:0.8rem; text-align:center; padding:1rem;">No active processes</div>';
                 return;
            }

            // Sort: Running first, then by time
            data.sort((a,b) => (a.state === 'running' ? -1 : 1));

            grid.innerHTML = data.map(s => {
                const isRunning = s.state === 'running';
                const badgeClass = s.state === 'succeeded' || s.state === 'completed' ? 'green' : s.state === 'failed' ? 'red' : s.state === 'running' ? 'blue' : '';
                
                return `
                <div class="status-card" style="border-left: 3px solid ${s.state === 'failed' ? 'var(--danger)' : s.state === 'succeeded' ? 'var(--success)' : 'var(--primary)'}">
                    <div class="left">
                         <div style="display:flex; align-items:center; gap:0.5rem">
                             <span class="sym">${s.symbol}</span>
                             <span class="badge ${badgeClass}">${s.state}</span>
                         </div>
                         <div class="desc">${s.description || '-'}</div>
                         ${s.error_message ? `<div style="color: var(--danger); font-size: 0.75rem">${s.error_message}</div>` : ''}
                    </div>
                    <div class="right">
                        ${isRunning ? `<button class="sm danger" onclick="stopTask('${s.symbol}')">Stop</button>` : ''}
                    </div>
                </div>
            `}).join('');
        }
        
        async function refreshLogs() {
            const res = await fetch('/logs?limit=50');
            const data = await res.json();
            const box = $('logs-box');
            // Check if scroll is at bottom
            const isAtBottom = box.scrollHeight - box.clientHeight <= box.scrollTop + 50;
            
            box.innerHTML = data.map(l => {
                 const cls = l.level === 'ERROR' ? 'log-error' : l.level === 'WARNING' ? 'log-warning' : 'log-info';
                 const ts = l.ts ? l.ts.split('T')[1].split('.')[0] : '';
                 return `<div class="log-entry ${cls}">
                    <span style="opacity:0.6; font-size:0.7em; margin-right:0.5rem;">${ts}</span>
                    <span style="opacity:0.8">[${l.level}]</span> ${l.message}
                 </div>`;
            }).join('');
            
            if (isAtBottom) box.scrollTop = box.scrollHeight;
        }

        async function loadDebugToggle() {
          const res = await fetch("/debug/alpaca/raw");
          const data = await res.json();
          $('alpaca-debug').checked = !!data.alpaca_debug_raw;
        }

        async function toggleAlpacaDebug(enabled) {
          await fetch(`/debug/alpaca/raw?enabled=${enabled}`, { method: "POST" });
          loadDebugToggle();
        }
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

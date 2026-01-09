from __future__ import annotations
import asyncio
import logging
from typing import Optional
import tempfile
import shutil
import duckdb
import pandas as pd

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .pipeline import run_pipeline, list_symbols as list_symbols_from_db

cfg = Config()
cfg.ensure_paths()
logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
logger = logging.getLogger("feature_service.web")
app = FastAPI(title="Feature Builder Service")


def _parquet_sample(symbol: str, limit: int) -> list[dict]:
  sym_root = cfg.dest_parquet / symbol
  if not sym_root.exists():
    logger.warning("parquet path missing for symbol", extra={"symbol": symbol, "parquet_root": str(sym_root)})
    return []
  # Pick the latest dated partition and read from its parquet file
  dt_dirs = sorted([p for p in sym_root.glob("dt=*") if p.is_dir()], reverse=True)
  for dt_dir in dt_dirs:
    parquet_files = list(dt_dir.glob("*.parquet"))
    if not parquet_files:
      continue
    try:
      df = pd.read_parquet(parquet_files[0])
      if df.empty:
        continue
      df = df.sort_values("ts", ascending=False).head(limit)
      return df.to_dict(orient="records")
    except Exception as exc:  # pragma: no cover
      logger.warning("failed to read parquet sample", extra={"symbol": symbol, "partition": str(dt_dir), "error": str(exc)})
      continue
  logger.info("no parquet data for symbol", extra={"symbol": symbol})
  return []

status: dict[str, object] = {
    "state": "idle",
    "symbols": [],
    "started_at": None,
    "completed_at": None,
    "result": None,
    "error": None,
}
status_lock = asyncio.Lock()


async def _run(symbols: Optional[list[str]], options: Optional[dict] = None) -> None:
    async with status_lock:
        status.update(
            {
                "state": "running",
                "symbols": symbols or "all",
                "started_at": datetime.now(tz=timezone.utc).isoformat(),
                "completed_at": None,
                "result": None,
                "error": None,
            }
        )
    try:
        result = await asyncio.to_thread(run_pipeline, cfg.source_db, cfg.dest_db, cfg.dest_parquet, symbols, options)
        async with status_lock:
            status.update(
                {
                    "state": "succeeded",
                    "completed_at": datetime.now(tz=timezone.utc).isoformat(),
                    "result": result,
                }
            )
    except Exception as exc:  # pragma: no cover
        logger.exception("feature run failed")
        async with status_lock:
            status.update(
                {
                    "state": "failed",
                    "error": str(exc),
                    "completed_at": datetime.now(tz=timezone.utc).isoformat(),
                }
            )


@app.get("/", response_class=HTMLResponse)
async def index():
    html = r"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Feature Builder</title>
      <style>
        body { font-family: 'Inter', system-ui, sans-serif; margin: 0; background: #0b1224; color: #e2e8f0; font-size: 0.9rem; }
        header { padding: 0.75rem 1rem; background: #111827; display:flex; justify-content: space-between; align-items:center; border-bottom: 1px solid #1f2937; }
        h1 { margin: 0; font-size: 1.1rem; }
        main { padding: 1rem; display: grid; gap: 0.75rem; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
        section { background: #111827; border-radius: 8px; padding: 0.75rem; border: 1px solid #1f2937; }
        section.full-width { grid-column: 1 / -1; }
        h3 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #94a3b8; }
        button { background: #2563eb; color: white; border: none; border-radius: 4px; padding: 0.35rem 0.6rem; cursor: pointer; font-size: 0.85rem; }
        button:hover { background: #1d4ed8; }
        button.ghost { background: #374151; }
        button.ghost:hover { background: #4b5563; }
        table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
        th, td { border-bottom: 1px solid #1f2937; padding: 0.35rem; text-align: left; }
        th { color: #94a3b8; font-weight: 600; }
        .badge { background: #1f2937; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.75rem; color: #94a3b8; }
        .pass { background: #064e3b; color: #6ee7b7; }
        .fail { background: #7f1d1d; color: #fca5a5; }
        pre { background: #0f172a; border-radius: 4px; padding: 0.5rem; max-height: 150px; overflow: auto; font-size: 0.75rem; margin: 0; }
        .row { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem; }
        .split { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
        input[type="checkbox"] { accent-color: #2563eb; }
      </style>
    </head>
    <body>
      <header>
        <h1>Feature Builder</h1>
        <div class="row" style="margin:0"> <span class="badge">Src: __SOURCE_DB__</span> <span class="badge">Dst: __DEST_DB__</span></div>
      </header>
      <main>
        <section>
          <h3>1. Load Tickers</h3>
          <div class="row">
            <button onclick="loadSymbols()">Load</button>
            <span id="step1-badge" class="badge">pending</span>
          </div>
          <div style="max-height: 200px; overflow-y: auto; border: 1px solid #1f2937; border-radius: 4px;">
            <table id="symbols-table"><thead><tr><th width="30"><input type="checkbox" onclick="toggleAll(this)"></th><th>Symbol</th></tr></thead><tbody></tbody></table>
          </div>
        </section>

        <section>
          <h3>2. Configure & Generate</h3>
          <div style="margin-bottom: 0.5rem; border: 1px solid #1f2937; padding: 0.5rem; border-radius: 4px;">
            <div class="row">
                <label><input type="checkbox" id="opt-sma" checked> SMA</label>
                <label><input type="checkbox" id="opt-bb" checked> Bollinger</label>
                <label><input type="checkbox" id="opt-rsi" checked> RSI</label>
                <label><input type="checkbox" id="opt-macd" checked> MACD</label>
                <label><input type="checkbox" id="opt-atr" checked> ATR</label>
                <label><input type="checkbox" id="opt-vol" checked> Volume</label>
                <label><input type="checkbox" id="opt-time" checked> Time</label>
            </div>
            <hr style="border: 0; border-top: 1px solid #1f2937; margin: 0.5rem 0;">
            <div class="row">
                <label><input type="checkbox" id="opt-segmentation" onchange="toggleSeg(this)"> Episode Mode</label>
            </div>
            <div id="seg-options" class="row" style="display:none; margin-left: 1rem;">
                <label>Train: <input type="number" id="opt-train" value="30" style="width: 50px; background: #1f2937; color: white; border: 1px solid #374151;"></label>
                <label>Test: <input type="number" id="opt-test" value="5" style="width: 50px; background: #1f2937; color: white; border: 1px solid #374151;"></label>
            </div>
          </div>
          <div class="row">
            <button onclick="runSelected()">Run Selected</button>
            <span id="step3-badge" class="badge">idle</span>
          </div>
          <div class="split">
             <div><small>Status</small><pre id="status-json">{}</pre></div>
             <div><small>Result</small><pre id="result-box">{}</pre></div>
          </div>
        </section>

        <section class="full-width">
          <h3>Database Viewer (feature_bars)</h3>
          <div class="row">
            <button class="ghost" onclick="firstPage()">First</button>
            <button class="ghost" onclick="prevPage()">Prev</button>
            <span id="page-info" class="badge">0-0 of 0</span>
            <button class="ghost" onclick="nextPage()">Next</button>
            <button class="ghost" onclick="lastPage()">Last</button>
            <input type="text" id="filter-symbol" placeholder="Symbol..." style="background: #1f2937; color: white; border: 1px solid #374151; padding: 0.25rem 0.5rem; border-radius: 4px; width: 80px; margin: 0 0.5rem;" onchange="loadRows(0)">
            <button onclick="loadRows(pageState.offset)">Refresh</button>
            <button class="fail" style="background: #7f1d1d; color: #fca5a5;" onclick="deleteAll()">Delete All</button>
          </div>
          <div style="overflow-x: auto;">
            <table id="features-table"><thead></thead><tbody></tbody></table>
          </div>
        </section>
      </main>
      <script>
        let pageState = { offset: 0, limit: 15, total: 0 };

        function setBadge(id, state, text) {
          const el = document.getElementById(id);
          if (!el) return;
          el.className = 'badge ' + (state === 'pass' ? 'pass' : state === 'fail' ? 'fail' : '');
          el.innerText = text || state;
        }

        async function deleteAll() {
            if (!confirm("Are you sure you want to delete ALL data? This will drop the table and clear parquet files.")) return;
            try {
                const res = await fetch('/features', { method: 'DELETE' });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                alert("Success: " + JSON.stringify(data));
                loadRows(0);
            } catch (e) {
                alert("Error: " + e.message);
            }
        }

        async function loadSymbols() {
          try {
            const res = await fetch('/symbols');
            const data = await res.json();
            const tbody = document.querySelector('#symbols-table tbody');
            tbody.innerHTML = '';
            data.forEach(sym => {
              const tr = document.createElement('tr');
              tr.innerHTML = `<td><input type="checkbox" value="${sym}"></td><td>${sym}</td>`;
              tbody.appendChild(tr);
            });
            setBadge('step1-badge', 'pass', `${data.length} loaded`);
          } catch (err) {
            setBadge('step1-badge', 'fail', 'error');
          }
        }

        function toggleAll(source) {
            document.querySelectorAll('#symbols-table tbody input').forEach(cb => cb.checked = source.checked);
        }

        function toggleSeg(cb) {
            document.getElementById('seg-options').style.display = cb.checked ? 'flex' : 'none';
        }

        async function runSelected() {
          const symbols = Array.from(document.querySelectorAll('#symbols-table tbody input:checked')).map(cb => cb.value);
          
          const options = {
              use_sma: document.getElementById('opt-sma').checked,
              use_bb: document.getElementById('opt-bb').checked,
              use_rsi: document.getElementById('opt-rsi').checked,
              use_macd: document.getElementById('opt-macd').checked,
              use_atr: document.getElementById('opt-atr').checked,
              use_vol: document.getElementById('opt-vol').checked,
              use_time: document.getElementById('opt-time').checked,
              enable_segmentation: document.getElementById('opt-segmentation').checked,
              train_window: parseInt(document.getElementById('opt-train').value),
              test_window: parseInt(document.getElementById('opt-test').value)
          };

          const payload = { symbols: symbols.length ? symbols : null, options: options };
          try {
            await fetch('/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            setBadge('step3-badge', 'pass', 'queued');
            pollStatus();
          } catch (err) {
            setBadge('step3-badge', 'fail', 'error');
          }
        }

        async function pollStatus() {
          const res = await fetch('/status');
          const data = await res.json();
          document.getElementById('status-json').innerText = JSON.stringify(data, null, 2);
          if (data.result) document.getElementById('result-box').innerText = JSON.stringify(data.result, null, 2);
          
          if (data.state === 'running') setTimeout(pollStatus, 1000);
          else if (data.state === 'succeeded') {
             setBadge('step3-badge', 'pass', 'done');
             loadRows(0);
          }
        }

        async function loadRows(offset) {
            if (offset < 0) offset = 0;
            try {
                const symFilter = document.getElementById('filter-symbol').value;
                const qs = `limit=${pageState.limit}&offset=${offset}` + (symFilter ? `&symbol=${symFilter}` : '');
                const res = await fetch(`/features/rows?${qs}`);
                const data = await res.json();
                pageState.offset = data.offset;
                pageState.total = data.total;
                
                const thead = document.querySelector('#features-table thead');
                const tbody = document.querySelector('#features-table tbody');
                thead.innerHTML = '';
                tbody.innerHTML = '';

                if (data.rows && data.rows.length > 0) {
                    const cols = Object.keys(data.rows[0]);
                    const trH = document.createElement('tr');
                    cols.forEach(c => {
                        const th = document.createElement('th');
                        th.innerText = c;
                        trH.appendChild(th);
                    });
                    thead.appendChild(trH);

                    data.rows.forEach(r => {
                        const tr = document.createElement('tr');
                        cols.forEach(c => {
                            const td = document.createElement('td');
                            if (c === 'options') {
                                const details = document.createElement('details');
                                const summary = document.createElement('summary');
                                summary.innerText = '{...}';
                                summary.style.cursor = 'pointer';
                                summary.style.color = '#94a3b8';
                                details.appendChild(summary);
                                const pre = document.createElement('pre');
                                pre.innerText = r[c];
                                pre.style.margin = '0.5rem 0 0 0';
                                pre.style.whiteSpace = 'pre-wrap';
                                details.appendChild(pre);
                                td.appendChild(details);
                            } else {
                                td.innerText = r[c];
                            }
                            tr.appendChild(td);
                        });
                        tbody.appendChild(tr);
                    });
                } else {
                    tbody.innerHTML = '<tr><td>No data</td></tr>';
                }
                
                const end = Math.min(pageState.offset + pageState.limit, pageState.total);
                document.getElementById('page-info').innerText = `${pageState.offset + 1}-${end} of ${pageState.total}`;

            } catch (e) {
                console.error(e);
            }
        }

        function firstPage() { loadRows(0); }
        function prevPage() { loadRows(pageState.offset - pageState.limit); }
        function nextPage() { if (pageState.offset + pageState.limit < pageState.total) loadRows(pageState.offset + pageState.limit); }
        function lastPage() { loadRows(Math.max(0, pageState.total - pageState.limit)); }

        loadSymbols();
        pollStatus();
        loadRows(0);
      </script>
    </body>
    </html>
    """
    html = html.replace("__SOURCE_DB__", str(cfg.source_db)).replace("__DEST_DB__", str(cfg.dest_db))
    return HTMLResponse(content=html)


@app.delete("/features")
async def delete_features():
    async with status_lock:
        if status["state"] == "running":
             raise HTTPException(status_code=409, detail="Pipeline is running")
        
        try:
            # Drop table from DuckDB
            conn = duckdb.connect(str(cfg.dest_db))
            conn.execute("DROP TABLE IF EXISTS feature_bars")
            conn.close()
            
            # Clear parquet directory
            if cfg.dest_parquet.exists():
                shutil.rmtree(cfg.dest_parquet)
                cfg.dest_parquet.mkdir(parents=True, exist_ok=True)
                
            return {"status": "deleted all data"}
        except Exception as e:
            logger.exception("failed to delete data")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/rows")
async def get_feature_rows(limit: int = 20, offset: int = 0, symbol: Optional[str] = None):
    conn = None
    tmpdir = None
    try:
        tmpdir = tempfile.TemporaryDirectory()
        tmp_dest = Path(tmpdir.name) / cfg.dest_db.name
        if not cfg.dest_db.exists():
             return {"rows": [], "total": 0, "offset": offset, "limit": limit}
        
        shutil.copy2(cfg.dest_db, tmp_dest)
        wal_src = cfg.dest_db.with_suffix(cfg.dest_db.suffix + ".wal")
        if wal_src.exists():
            wal_dst = tmp_dest.with_suffix(tmp_dest.suffix + ".wal")
            shutil.copy2(wal_src, wal_dst)
        
        conn = duckdb.connect(str(tmp_dest), read_only=True)
        
        try:
            conn.execute("SELECT 1 FROM feature_bars LIMIT 1")
        except duckdb.Error:
             return {"rows": [], "total": 0, "offset": offset, "limit": limit}

        if symbol:
            total = conn.execute("SELECT COUNT(*) FROM feature_bars WHERE symbol = ?", [symbol]).fetchone()[0]
            df = conn.execute(
                "SELECT * FROM feature_bars WHERE symbol = ? ORDER BY ts DESC LIMIT ? OFFSET ?",
                [symbol, limit, offset]
            ).fetch_df()
        else:
            total = conn.execute("SELECT COUNT(*) FROM feature_bars").fetchone()[0]
            df = conn.execute(
                "SELECT * FROM feature_bars ORDER BY ts DESC LIMIT ? OFFSET ?",
                [limit, offset]
            ).fetch_df()
        
        df = df.fillna("")
        for col in df.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns:
            df[col] = df[col].astype(str)

        return {
            "rows": df.to_dict(orient="records"),
            "total": total,
            "offset": offset,
            "limit": limit
        }
    except Exception as e:
        logger.error("failed to fetch rows", extra={"error": str(e)})
        return {"rows": [], "total": 0, "offset": offset, "limit": limit, "error": str(e)}
    finally:
        if conn:
            conn.close()
        if tmpdir:
            tmpdir.cleanup()


@app.get("/symbols")
async def symbols():
    conn = None
    tmpdir = None
    try:
        if not cfg.source_db.exists():
            logger.warning("source db not found", extra={"path": str(cfg.source_db)})
            return []

        tmpdir = tempfile.TemporaryDirectory()
        tmp_src = Path(tmpdir.name) / cfg.source_db.name
        shutil.copy2(cfg.source_db, tmp_src)
        wal_src = cfg.source_db.with_suffix(cfg.source_db.suffix + ".wal")
        if wal_src.exists():
            wal_dst = tmp_src.with_suffix(tmp_src.suffix + ".wal")
            shutil.copy2(wal_src, wal_dst)
        conn = __import__("duckdb").connect(str(tmp_src), read_only=True)
        return list_symbols_from_db(conn)
    except Exception as e:
        logger.exception("failed to list symbols")
        return []
    finally:
        if conn:
            conn.close()
        if tmpdir:
            tmpdir.cleanup()


@app.get("/features_sample")
async def features_sample(symbol: str, limit: int = 3):
    conn = None
    tmpdir = None
    try:
        tmpdir = tempfile.TemporaryDirectory()
        tmp_dest = Path(tmpdir.name) / cfg.dest_db.name
        shutil.copy2(cfg.dest_db, tmp_dest)
        wal_src = cfg.dest_db.with_suffix(cfg.dest_db.suffix + ".wal")
        if wal_src.exists():
            wal_dst = tmp_dest.with_suffix(tmp_dest.suffix + ".wal")
            shutil.copy2(wal_src, wal_dst)
        conn = duckdb.connect(str(tmp_dest), read_only=True)
        # Return from DuckDB when the table and rows exist; otherwise fall back to parquet partitions
        try:
          conn.execute("SELECT 1 FROM feature_bars LIMIT 1")
        except duckdb.Error:
          logger.warning("feature_bars table missing in dest db", extra={"dest_db": str(cfg.dest_db)})
          return _parquet_sample(symbol, limit)

        df = conn.execute(
          "SELECT * FROM feature_bars WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
          [symbol, limit],
        ).fetch_df()
        if not df.empty:
          return df.to_dict(orient="records")
        logger.info("no feature_bars rows for symbol; checking parquet", extra={"symbol": symbol})
        return _parquet_sample(symbol, limit)
    finally:
        if conn:
            conn.close()
        if tmpdir:
            tmpdir.cleanup()


class RunRequest(BaseModel):
    symbols: Optional[list[str]] = None
    options: Optional[dict] = None


@app.post("/run")
async def run_features(request: RunRequest, background: BackgroundTasks):
    if status["state"] == "running":
        raise HTTPException(status_code=409, detail="run already in progress")
    symbols = request.symbols
    options = request.options
    background.add_task(_run, symbols, options)
    return {"state": "queued"}


@app.get("/status")
async def get_status():
    async with status_lock:
        return dict(status)


@app.on_event("startup")
async def log_routes() -> None:
    route_paths = [getattr(route, "path", "?") for route in app.routes]
    logger.info("registered routes", extra={"routes": route_paths})

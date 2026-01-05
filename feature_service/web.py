from __future__ import annotations
import asyncio
import logging
from typing import Optional
import tempfile
import shutil
import duckdb

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

status: dict[str, object] = {
    "state": "idle",
    "symbols": [],
    "started_at": None,
    "completed_at": None,
    "result": None,
    "error": None,
}
status_lock = asyncio.Lock()


async def _run(symbols: Optional[list[str]]) -> None:
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
        result = await asyncio.to_thread(run_pipeline, cfg.source_db, cfg.dest_db, cfg.dest_parquet, symbols)
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
        body { font-family: 'Inter', system-ui, sans-serif; margin: 0; background: #0b1224; color: #e2e8f0; }
        header { padding: 1rem 1.5rem; background: #111827; display:flex; justify-content: space-between; align-items:center; }
        h1 { margin: 0; font-size: 1.2rem; }
        main { padding: 1rem 1.5rem; display: grid; gap: 1rem; }
        section { background: #111827; border-radius: 12px; padding: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,0.35); }
        button { background: #22c55e; color: #0b1224; border: none; border-radius: 8px; padding: 0.5rem 0.75rem; cursor: pointer; font-weight: 700; }
        button:hover { background: #16a34a; }
        .ghost { background: #1f2937; color: #e2e8f0; }
        table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
        th, td { border-bottom: 1px solid #1f2937; padding: 0.5rem; font-size: 0.95rem; }
        .badge { background: #1f2937; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.85rem; }
        .pass { background: #14532d; color: #bef264; }
        .fail { background: #7f1d1d; color: #fecaca; }
        .pending { background: #1f2937; color: #e2e8f0; }
        pre { background: #0b172f; border-radius: 10px; padding: 0.75rem; max-height: 320px; overflow: auto; }
        .row { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
        a { color: #38bdf8; }
        .step { display: grid; gap: 0.5rem; }
        .split { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.5rem; }
      </style>
    </head>
    <body>
      <header>
        <h1>Feature Builder</h1>
        <div class="row"> <span class="badge">Source DB: __SOURCE_DB__</span> <span class="badge">Dest DB: __DEST_DB__</span></div>
      </header>
      <main>
        <section class="step">
          <h3>Step 1: Load unique tickers</h3>
          <div class="row">
            <button onclick="loadSymbols()">Test: load tickers</button>
            <span id="step1-badge" class="badge pending">pending</span>
          </div>
          <div class="split">
            <div>
              <strong>Raw before</strong>
              <pre id="step1-before">[]</pre>
            </div>
            <div>
              <strong>Raw after</strong>
              <pre id="step1-after">[]</pre>
            </div>
          </div>
          <table id="symbols-table"><thead><tr><th></th><th>Symbol</th></tr></thead><tbody></tbody></table>
        </section>

        <section class="step">
          <h3>Step 2: Select tickers</h3>
          <div class="row">
            <button class="ghost" onclick="selectAll()">Select all</button>
            <button class="ghost" onclick="clearSelection()">Clear</button>
            <button onclick="testSelection()">Test selection</button>
            <span id="step2-badge" class="badge pending">pending</span>
          </div>
          <div class="split">
            <div>
              <strong>Raw before</strong>
              <pre id="step2-before">[]</pre>
            </div>
            <div>
              <strong>Raw after</strong>
              <pre id="step2-after">[]</pre>
            </div>
          </div>
        </section>

        <section class="step">
          <h3>Step 3: Generate features</h3>
          <div class="row">
            <button onclick="runSelected()">Test: run feature generation</button>
            <span id="step3-badge" class="badge pending">pending</span>
          </div>
          <div class="split">
            <div>
              <strong>Raw before</strong>
              <pre id="step3-before">{ symbols: [] }</pre>
            </div>
            <div>
              <strong>Raw after</strong>
              <pre id="step3-after">{}</pre>
            </div>
          </div>
        </section>

        <section>
          <h3>Run status</h3>
          <div class="row"><span id="status-badge" class="badge pending">idle</span></div>
          <div id="status-json" class="badge"></div>
          <details open style="margin-top:0.5rem;"><summary>Result</summary><pre id="result-box"></pre></details>
        </section>
      </main>
      <script>
        let lastSymbols = [];
        let lastSelection = [];
        let lastRunSymbols = [];

        function setBadge(id, state, text) {
          const el = document.getElementById(id);
          if (!el) return;
          el.classList.remove('pass', 'fail', 'pending');
          if (state === 'pass') {
            el.classList.add('pass');
            el.innerText = text || 'pass';
          } else if (state === 'fail') {
            el.classList.add('fail');
            el.innerText = text || 'fail';
          } else {
            el.classList.add('pending');
            el.innerText = text || 'pending';
          }
        }

        async function loadSymbols() {
          const beforePayload = { endpoint: '/symbols', previous: lastSymbols };
          document.getElementById('step1-before').innerText = JSON.stringify(beforePayload, null, 2);
          try {
            const res = await fetch('/symbols');
            if (!res.ok) {
              const text = await res.text();
              throw new Error('status ' + res.status + ' body: ' + text.slice(0, 400));
            }
            const data = await res.json();
            lastSymbols = data;
            document.getElementById('step1-after').innerText = JSON.stringify(data, null, 2);
            const tbody = document.querySelector('#symbols-table tbody');
            tbody.innerHTML = '';
            data.forEach(sym => {
              const tr = document.createElement('tr');
              tr.innerHTML = '<td><input type="checkbox" value="' + sym + '" /></td><td>' + sym + '</td>';
              tbody.appendChild(tr);
            });
            setBadge('step1-badge', 'pass', data.length ? 'pass' : 'empty');
          } catch (err) {
            document.getElementById('step1-after').innerText = String(err);
            setBadge('step1-badge', 'fail', 'fail');
          }
        }

        function selectAll() { document.querySelectorAll('#symbols-table tbody input').forEach(cb => cb.checked = true); }
        function clearSelection() { document.querySelectorAll('#symbols-table tbody input').forEach(cb => cb.checked = false); }

        function testSelection() {
          document.getElementById('step2-before').innerText = JSON.stringify({ selected: lastSelection }, null, 2);
          const all = Array.from(document.querySelectorAll('#symbols-table tbody input'));
          const selected = all.filter(cb => cb.checked).map(cb => cb.value);
          document.getElementById('step2-after').innerText = JSON.stringify(selected, null, 2);
          lastSelection = selected;
          if (all.length === 0) {
            setBadge('step2-badge', 'fail', 'no symbols loaded');
            return;
          }
          if (selected.length === 0) {
            setBadge('step2-badge', 'fail', 'no selection');
            return;
          }
          setBadge('step2-badge', 'pass', selected.length === all.length ? 'all selected' : 'partial');
        }

        async function runSelected() {
          const symbols = Array.from(document.querySelectorAll('#symbols-table tbody input:checked')).map(cb => cb.value);
          document.getElementById('step3-before').innerText = JSON.stringify({ symbols: symbols }, null, 2);
          lastRunSymbols = symbols;
          const payload = symbols.length ? { symbols: symbols } : {};
          try {
            const res = await fetch('/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            if (!res.ok) {
              const text = await res.text();
              throw new Error('status ' + res.status + ' body: ' + text.slice(0, 400));
            }
            const data = await res.json();
            document.getElementById('step3-after').innerText = JSON.stringify({ run: data, sample: 'pending' }, null, 2);
            setBadge('step3-badge', 'pass', 'queued');
            pollStatus();
          } catch (err) {
            document.getElementById('step3-after').innerText = String(err);
            setBadge('step3-badge', 'fail', 'fail');
          }
        }

        async function pollStatus() {
          const res = await fetch('/status');
          const data = await res.json();
          document.getElementById('status-badge').innerText = data.state;
          document.getElementById('status-json').innerText = JSON.stringify(data, null, 2);
          document.getElementById('result-box').innerText = JSON.stringify(data.result, null, 2);

          if (data.state === 'succeeded' && lastRunSymbols.length) {
            try {
              const sym = lastRunSymbols[0];
              const sampRes = await fetch('/features_sample?symbol=' + encodeURIComponent(sym) + '&limit=3');
              if (sampRes.ok) {
                const sample = await sampRes.json();
                document.getElementById('step3-after').innerText = JSON.stringify({ result: data.result, sample }, null, 2);
              } else {
                document.getElementById('step3-after').innerText = JSON.stringify({ result: data.result, sample_error: 'status ' + sampRes.status }, null, 2);
              }
            } catch (e) {
              document.getElementById('step3-after').innerText = JSON.stringify({ result: data.result, sample_error: String(e) }, null, 2);
            }
          }

          if (data.state === 'running') { setTimeout(pollStatus, 1500); }
        }

        loadSymbols();
        pollStatus();
      </script>
    </body>
    </html>
    """
    html = html.replace("__SOURCE_DB__", str(cfg.source_db)).replace("__DEST_DB__", str(cfg.dest_db))
    return HTMLResponse(content=html)


@app.get("/symbols")
async def symbols():
    conn = None
    tmpdir = None
    try:
        tmpdir = tempfile.TemporaryDirectory()
        tmp_src = Path(tmpdir.name) / cfg.source_db.name
        shutil.copy2(cfg.source_db, tmp_src)
        wal_src = cfg.source_db.with_suffix(cfg.source_db.suffix + ".wal")
        if wal_src.exists():
            wal_dst = tmp_src.with_suffix(tmp_src.suffix + ".wal")
            shutil.copy2(wal_src, wal_dst)
        conn = __import__("duckdb").connect(str(tmp_src), read_only=True)
        return list_symbols_from_db(conn)
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
        try:
          conn.execute("SELECT 1 FROM feature_bars LIMIT 1")
        except duckdb.Error:
          return []
        df = conn.execute(
          "SELECT * FROM feature_bars WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
          [symbol, limit],
        ).fetch_df()
        return df.to_dict(orient="records")
      finally:
        if conn:
          conn.close()
        if tmpdir:
          tmpdir.cleanup()


class RunRequest(BaseModel):
    symbols: Optional[list[str]] = None


@app.post("/run")
async def run_features(request: RunRequest, background: BackgroundTasks):
    if status["state"] == "running":
        raise HTTPException(status_code=409, detail="run already in progress")
    symbols = request.symbols
    background.add_task(_run, symbols)
    return {"state": "queued"}


@app.get("/status")
async def get_status():
    async with status_lock:
        return dict(status)

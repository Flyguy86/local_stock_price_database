from __future__ import annotations
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime, timezone

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
    html = f"""
    <!doctype html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <title>Feature Builder</title>
      <style>
        body {{ font-family: 'Inter', system-ui, sans-serif; margin: 0; background: #0b1224; color: #e2e8f0; }}
        header {{ padding: 1rem 1.5rem; background: #111827; display:flex; justify-content: space-between; align-items:center; }}
        h1 {{ margin: 0; font-size: 1.2rem; }}
        main {{ padding: 1rem 1.5rem; display: grid; gap: 1rem; }}
        section {{ background: #111827; border-radius: 12px; padding: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,0.35); }}
        button {{ background: #22c55e; color: #0b1224; border: none; border-radius: 8px; padding: 0.5rem 0.75rem; cursor: pointer; font-weight: 700; }}
        button:hover {{ background: #16a34a; }}
        .ghost {{ background: #1f2937; color: #e2e8f0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
        th, td {{ border-bottom: 1px solid #1f2937; padding: 0.5rem; font-size: 0.95rem; }}
        .badge {{ background: #1f2937; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.85rem; }}
        pre {{ background: #0b172f; border-radius: 10px; padding: 0.75rem; max-height: 320px; overflow: auto; }}
        .row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }}
        a {{ color: #38bdf8; }}
      </style>
    </head>
    <body>
      <header>
        <h1>Feature Builder</h1>
        <div class=\"row\"> <span class=\"badge\">Source DB: {cfg.source_db}</span> <span class=\"badge\">Dest DB: {cfg.dest_db}</span></div>
      </header>
      <main>
        <section>
          <h3>Symbols</h3>
          <div class=\"row\">
            <button onclick=\"loadSymbols()\">Refresh list</button>
            <button class=\"ghost\" onclick=\"selectAll()\">Select all</button>
            <button class=\"ghost\" onclick=\"clearSelection()\">Clear</button>
            <button onclick=\"runSelected()\">Generate features</button>
            <span id=\"status-badge\" class=\"badge\"></span>
          </div>
          <table id=\"symbols-table\"><thead><tr><th></th><th>Symbol</th></tr></thead><tbody></tbody></table>
        </section>
        <section>
          <h3>Run status</h3>
          <div id=\"status-json\" class=\"badge\"></div>
          <details open style=\"margin-top:0.5rem;\"><summary>Result</summary><pre id=\"result-box\"></pre></details>
        </section>
      </main>
      <script>
        async function loadSymbols() {{
          const res = await fetch('/symbols');
          const data = await res.json();
          const tbody = document.querySelector('#symbols-table tbody');
          tbody.innerHTML = '';
          data.forEach(sym => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `<td><input type="checkbox" value="${{sym}}" /></td><td>${{sym}}</td>`;
            tbody.appendChild(tr);
          }});
        }}

        function selectAll() {{ document.querySelectorAll('#symbols-table tbody input').forEach(cb => cb.checked = true); }}
        function clearSelection() {{ document.querySelectorAll('#symbols-table tbody input').forEach(cb => cb.checked = false); }}

        async function runSelected() {{
          const symbols = Array.from(document.querySelectorAll('#symbols-table tbody input:checked')).map(cb => cb.value);
          const payload = symbols.length ? {{ symbols }} : {{}};
          const res = await fetch('/run', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(payload) }});
          if (!res.ok) {{ alert('failed to start run'); return; }}
          pollStatus();
        }}

        async function pollStatus() {{
          const res = await fetch('/status');
          const data = await res.json();
          document.getElementById('status-badge').innerText = data.state;
          document.getElementById('status-json').innerText = JSON.stringify(data, null, 2);
          document.getElementById('result-box').innerText = JSON.stringify(data.result, null, 2);
          if (data.state === 'running') {{ setTimeout(pollStatus, 1500); }}
        }}

        loadSymbols();
        pollStatus();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/symbols")
async def symbols():
    conn = None
    try:
        conn = __import__("duckdb").connect(str(cfg.source_db))
        return list_symbols_from_db(conn)
    finally:
        if conn:
            conn.close()


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

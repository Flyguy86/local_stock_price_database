from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
import uvicorn
import logging
import sys
from pathlib import Path

# Add shared modules to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from optimization_service import database
# Import simulation core just to get lists of models/tickers for the UI form
from simulation_service.core import get_available_models, get_available_tickers

app = FastAPI(title="Optimization Service C2")
log = logging.getLogger("optimization.server")

@app.on_event("startup")
def startup():
    database.ensure_tables()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Optimization C2</title>
    <style>
        body { font-family: sans-serif; padding: 20px; background: #f0f2f5; }
        .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        .stat-box { display: inline-block; padding: 15px; background: #e3f2fd; border-radius: 5px; margin-right: 10px; }
        .stat-val { font-size: 24px; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .status-running { color: orange; font-weight: bold; }
        .status-completed { color: green; font-weight: bold; }
        .btn { padding: 10px 15px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
    </style>
    <script>
        async function refreshStats() {
            const res = await fetch('/api/stats');
            const data = await res.json();
            
            const counts = data.counts;
            document.getElementById('s-pending').innerText = counts.PENDING || 0;
            document.getElementById('s-running').innerText = counts.RUNNING || 0;
            document.getElementById('s-completed').innerText = counts.COMPLETED || 0;
            document.getElementById('s-failed').innerText = counts.FAILED || 0;
            
            // Leaderboard
            const lb = document.getElementById('leaderboard-body');
            lb.innerHTML = '';
            data.leaderboard.forEach(row => {
               lb.innerHTML += `<tr>
                   <td>${row.ticker}</td>
                   <td>${row.model}</td>
                   <td><b>${row.return.toFixed(2)}%</b></td>
                   <td>${row.trades}</td>
                   <td><small>${JSON.stringify(row.params)}</small></td>
               </tr>`; 
            });
        }
        
        async function createJob() {
            const models = Array.from(document.getElementById('models').selectedOptions).map(o => o.value);
            const tickers = Array.from(document.getElementById('tickers').selectedOptions).map(o => o.value);
            
            const payload = {
                models: models,
                tickers: tickers,
                thresholds: [0.0, 0.001, 0.002], // Default grid
                z_score: [true, false],
                use_bot: [false] 
            };
            
            await fetch('/api/create_batch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            alert('Batch created!');
            refreshStats();
        }
        
        setInterval(refreshStats, 3000);
        window.onload = refreshStats;
    </script>
</head>
<body>
    <h1>Optimization Command & Control</h1>
    
    <div class="card">
        <h2>Cluster Status</h2>
        <div class="stat-box">Pending: <div id="s-pending" class="stat-val">0</div></div>
        <div class="stat-box">Running: <div id="s-running" class="stat-val">0</div></div>
        <div class="stat-box">Completed: <div id="s-completed" class="stat-val">0</div></div>
        <div class="stat-box">Failed: <div id="s-failed" class="stat-val">0</div></div>
    </div>

    <div class="card">
        <h2>Launch Grid Search</h2>
        <form onsubmit="event.preventDefault(); createJob();">
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <label>Models</label><br>
                    <select id="models" multiple style="width:100%; height:150px">
                        {model_opts}
                    </select>
                </div>
                <div>
                    <label>Tickers</label><br>
                    <select id="tickers" multiple style="width:100%; height:150px">
                        {ticker_opts}
                    </select>
                </div>
            </div>
            <br>
            <button class="btn">Queue Batch</button>
        </form>
    </div>

    <div class="card">
        <h2>Top Performers</h2>
        <table>
            <thead><tr><th>Ticker</th><th>Model</th><th>Return</th><th>Trades</th><th>Params</th></tr></thead>
            <tbody id="leaderboard-body"></tbody>
        </table>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    models = get_available_models()
    tickers = get_available_tickers()
    
    m_opts = "".join([f"<option value='{m['id']}'>{m['name']}</option>" for m in models])
    t_opts = "".join([f"<option value='{t}'>{t}</option>" for t in tickers])
    
    return HTML_TEMPLATE.format(model_opts=m_opts, ticker_opts=t_opts)

@app.get("/api/stats")
def get_stats():
    return database.get_dashboard_stats()

@app.post("/api/create_batch")
async def create_batch(config: dict):
    # Cross product generation
    jobs = []
    models = config.get("models", [])
    tickers = config.get("tickers", [])
    thresholds = config.get("thresholds", [0.0])
    z_scores = config.get("z_score", [False])
    
    for m in models:
        for t in tickers:
            for thresh in thresholds:
                for z in z_scores:
                    jobs.append({
                        "model_id": m,
                        "ticker": t,
                        "initial_cash": 10000,
                        "use_bot": False,
                        "min_prediction_threshold": thresh,
                        "enable_z_score_check": z,
                        "volatility_normalization": False
                    })
    
    batch_id = database.create_jobs(jobs)
    return {"batch_id": batch_id, "jobs_created": len(jobs)}

# --- Worker API ---

@app.post("/api/worker/claim")
def claim_job(payload: dict):
    worker_id = payload.get("worker_id", "unknown")
    job = database.claim_job(worker_id)
    return job # Returns null if empty

@app.post("/api/worker/complete")
def complete_job(payload: dict):
    job_id = payload.get("job_id")
    result = payload.get("result")
    status = payload.get("status", "COMPLETED")
    database.complete_job(job_id, result, status)
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

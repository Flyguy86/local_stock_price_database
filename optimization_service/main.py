from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
import sys
from pathlib import Path
import threading

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from optimization_service import database
from simulation_service.core import get_available_models, get_available_tickers

log = logging.getLogger("optimization.server")

# Import worker logic
from optimization_service.worker import run_worker

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    database.ensure_tables()
    log.info("Optimization C2 startup complete")
    
    # Start internal worker thread (for testing/single-node deployment)
    try:
        log.info("Attempting to start internal worker thread...")
        worker_thread = threading.Thread(target=run_worker, daemon=True, name="InternalWorker")
        worker_thread.start()
        log.info(f"✓ Started internal worker thread (daemon={worker_thread.daemon}, alive={worker_thread.is_alive()})")
        
        # Give worker a moment to initialize
        import time
        time.sleep(1)
        
        if worker_thread.is_alive():
            log.info("✓ Worker thread is running and healthy")
        else:
            log.error("✗ Worker thread died immediately after start")
            
    except Exception as e:
        log.error(f"✗ Failed to start worker thread: {e}", exc_info=True)
    
    yield
    
    # Shutdown
    log.info("Optimization C2 shutdown")

app = FastAPI(title="Optimization Service C2", lifespan=lifespan)

# HTML template - note: {{ and }} are used for CSS/JS, __PLACEHOLDERS__ for Python
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Optimization C2 - Strategy Heatmap</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: #0a0e27; color: #e0e0e0; }
        .card { background: #1a1f3a; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #2a3a5a; }
        h1 { color: #4fc3f7; font-weight: 300; }
        h2 { color: #81c784; font-size: 18px; border-bottom: 2px solid #2a3a5a; padding-bottom: 8px; }
        .stat-box { display: inline-block; padding: 15px 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; margin-right: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        .stat-val { font-size: 32px; font-weight: bold; color: white; }
        .stat-label { font-size: 11px; color: rgba(255,255,255,0.8); text-transform: uppercase; letter-spacing: 1px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #2a3a5a; }
        th { background-color: #2a3a5a; color: #81c784; font-weight: 600; }
        tr:hover { background-color: #252b4a; }
        .btn { padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.3s; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
        .worker-box { padding: 12px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-left: 4px solid #0fb8ad; margin-bottom: 10px; border-radius: 6px; color: white; }
        .worker-idle { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-left-color: #f5576c; }
        .progress-bar { width: 100%; height: 24px; background: rgba(255,255,255,0.1); border-radius: 12px; overflow: hidden; margin-top: 6px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%); transition: width 0.3s; box-shadow: 0 0 10px rgba(0,210,255,0.5); }
        select { width: 100%; padding: 10px; background: #2a3a5a; color: #e0e0e0; border: 1px solid #3a4a6a; border-radius: 4px; font-size: 13px; }
        option { background: #1a1f3a; }
        label { display: block; margin-bottom: 6px; color: #81c784; font-weight: 600; font-size: 13px; }
        .grid-config { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px; }
        .config-section { background: #252b4a; padding: 15px; border-radius: 6px; }
        .config-section h3 { margin-top: 0; color: #4fc3f7; font-size: 14px; }
        input[type="number"] { width: 60px; padding: 6px; background: #1a1f3a; color: #e0e0e0; border: 1px solid #3a4a6a; border-radius: 4px; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
        .badge-success { background: #4caf50; color: white; }
        .badge-warning { background: #ff9800; color: white; }
        .badge-danger { background: #f44336; color: white; }
        .heatmap-hint { background: #2a3a5a; padding: 12px; border-radius: 6px; margin-top: 15px; font-size: 12px; color: #b0bec5; border-left: 3px solid #4fc3f7; }
        .params-toggle { cursor: pointer; color: #4fc3f7; font-size: 11px; text-decoration: underline; }
        .params-toggle:hover { color: #81c784; }
        .params-json { display: none; margin-top: 8px; padding: 10px; background: #1a1f3a; border-left: 3px solid #4fc3f7; border-radius: 4px; font-family: monospace; font-size: 10px; white-space: pre-wrap; word-wrap: break-word; max-width: 300px; }
        .params-json.expanded { display: block; }
        .sim-methods { display: none; margin-top: 8px; padding: 10px; background: #1a1f3a; border-left: 3px solid #ffa726; border-radius: 4px; font-family: monospace; font-size: 10px; white-space: pre-wrap; word-wrap: break-word; max-width: 400px; line-height: 1.6; }
        .sim-methods.expanded { display: block; }
    </style>
    <script>
        const expandedParams = new Set();
        const expandedSimMethods = new Set();
        
        function escapeHtml(unsafe) {
            if (!unsafe) return '';
            return unsafe.toString()
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        // NEW: Helper to generate all non-empty subsets (combinations)
        function getCombinations(values) {
            let result = [];
            const f = function(prefix, values) {
                for (let i = 0; i < values.length; i++) {
                    let newPrefix = prefix.concat(values[i]);
                    result.push(newPrefix);
                    f(newPrefix, values.slice(i + 1));
                }
            };
            f([], values);
            return result;
        }
        
        async function refreshStats() {
            const res = await fetch('/api/stats');
            const data = await res.json();
            
            const counts = data.counts;
            document.getElementById('s-pending').innerText = counts.PENDING || 0;
            document.getElementById('s-running').innerText = counts.RUNNING || 0;
            document.getElementById('s-completed').innerText = counts.COMPLETED || 0;
            document.getElementById('s-failed').innerText = counts.FAILED || 0;
            
            const workersDiv = document.getElementById('workers-list');
            workersDiv.innerHTML = '';
            if (data.workers && data.workers.length > 0) {
                data.workers.forEach(w => {
                    const cls = w.current_job_id ? 'worker-box' : 'worker-box worker-idle';
                    const jobInfo = w.job_params 
                        ? escapeHtml(w.job_params.ticker) + ' | ' + escapeHtml(w.job_params.model_id.substring(0,12)) + ' | Thresh: ' + w.job_params.min_prediction_threshold
                        : 'Idle - Waiting for jobs';
                    const progressPct = (w.progress || 0) * 100;
                    
                    const div = document.createElement('div');
                    div.className = cls;
                    div.innerHTML = '<strong>Worker ' + escapeHtml(w.id) + '</strong>: ' + jobInfo + 
                        '<div class="progress-bar"><div class="progress-fill" style="width: ' + progressPct + '%"></div></div>';
                    workersDiv.appendChild(div);
                });
            } else {
                workersDiv.innerHTML = '<p style="color: #f5576c; text-align: center; padding: 20px;">No active workers detected. Launch agents: <code style="background: #2a3a5a; padding: 4px 8px; border-radius: 4px;">python optimization_service/worker.py</code></p>';
            }
            
            const lb = document.getElementById('leaderboard-body');
            lb.innerHTML = '';
            
            if (data.leaderboard && data.leaderboard.length > 0) {
                data.leaderboard.forEach((row, idx) => {
                    const returnClass = row.return > 0 ? 'badge-success' : 'badge-danger';
                    const hitClass = row.hit_rate > 55 ? 'badge-success' : (row.hit_rate > 50 ? 'badge-warning' : 'badge-danger');
                    const sqnClass = row.sqn > 3 ? 'badge-success' : (row.sqn > 2 ? 'badge-warning' : 'badge-danger');
                    
                    const rank = idx === 0 ? '1st' : (idx === 1 ? '2nd' : (idx === 2 ? '3rd' : (idx + 1) + 'th'));
                    
                    const zScore = row.z_score ? 'Y' : 'N';
                    const volNorm = row.vol_norm ? 'Y' : 'N';
                    const useBot = row.use_bot ? 'Y' : 'N';
                    
                    // CHANGED: Use stable row ID based on ticker + model + threshold
                    const rowId = 'row-' + row.ticker + '-' + row.model.replace(/\./g, '') + '-' + row.threshold.toFixed(4).replace('.', '');
                    const paramsJson = JSON.stringify(row.full_params, null, 2);
                    const simMethodsText = row.sim_methods || 'No simulation details available';
                    
                    const tr = document.createElement('tr');
                    tr.innerHTML = '<td><strong>' + rank + '</strong></td>' +
                        '<td>' + escapeHtml(row.ticker) + '</td>' +
                        '<td>' + escapeHtml(row.model) + '</td>' +
                        '<td><span class="badge ' + sqnClass + '">' + row.sqn.toFixed(2) + '</span></td>' +
                        '<td><span class="badge ' + returnClass + '">' + row.return.toFixed(2) + '%</span></td>' +
                        '<td><span class="badge ' + hitClass + '">' + (row.hit_rate ? row.hit_rate.toFixed(1) : 'N/A') + '%</span></td>' +
                        '<td>' + (row.trades || 'N/A') + '</td>' +
                        '<td>' + row.expectancy.toFixed(2) + '</td>' +
                        '<td>' + row.profit_factor.toFixed(2) + '</td>' +
                        '<td>' + row.threshold.toFixed(4) + '</td>' +
                        '<td>' + zScore + '</td>' +
                        '<td>' + volNorm + '</td>' +
                        '<td>' + useBot + '</td>' +
                        '<td style="font-size: 10px;">' + escapeHtml(row.regime_col || 'None') + '</td>' +
                        '<td style="font-size: 10px;">' + escapeHtml(row.allowed_regimes) + '</td>' +
                        '<td><span class="params-toggle" id="toggle-sim-' + rowId + '">Show Methods</span>' +
                        '<div class="sim-methods" id="sim-methods-' + rowId + '"></div></td>' +
                        '<td><span class="params-toggle" id="toggle-' + rowId + '">Show Params</span>' +
                        '<div class="params-json" id="params-' + rowId + '"></div></td>';
                    
                    // Sim Methods toggle
                    const simToggleSpan = tr.querySelector('#toggle-sim-' + rowId);
                    const simMethodsDiv = tr.querySelector('#sim-methods-' + rowId);
                    simMethodsDiv.textContent = simMethodsText;
                    
                    if (expandedSimMethods.has(rowId)) {
                        simMethodsDiv.classList.add('expanded');
                        simToggleSpan.innerText = 'Hide Methods';
                    }
                    
                    simToggleSpan.addEventListener('click', function() {
                        if (simMethodsDiv.classList.contains('expanded')) {
                            simMethodsDiv.classList.remove('expanded');
                            simToggleSpan.innerText = 'Show Methods';
                            expandedSimMethods.delete(rowId);
                        } else {
                            simMethodsDiv.classList.add('expanded');
                            simToggleSpan.innerText = 'Hide Methods';
                            expandedSimMethods.add(rowId);
                        }
                    });
                    
                    // Params toggle (existing)
                    const toggleSpan = tr.querySelector('#toggle-' + rowId);
                    const jsonDiv = tr.querySelector('#params-' + rowId);
                    jsonDiv.textContent = paramsJson;
                    
                    if (expandedParams.has(rowId)) {
                        jsonDiv.classList.add('expanded');
                        toggleSpan.innerText = 'Hide Params';
                    }
                    
                    toggleSpan.addEventListener('click', function() {
                        if (jsonDiv.classList.contains('expanded')) {
                            jsonDiv.classList.remove('expanded');
                            toggleSpan.innerText = 'Show Params';
                            expandedParams.delete(rowId);
                        } else {
                            jsonDiv.classList.add('expanded');
                            toggleSpan.innerText = 'Hide Params';
                            expandedParams.add(rowId);
                        }
                    });
                    
                    lb.appendChild(tr);
                });
            } else {
                lb.innerHTML = '<tr><td colspan="17" style="text-align: center; color: #f5576c;">No completed runs yet. Queue your first batch!</td></tr>';
            }
        }
        
        async function createJob() {
            const models = Array.from(document.getElementById('models').selectedOptions).map(o => o.value);
            const tickers = Array.from(document.getElementById('tickers').selectedOptions).map(o => o.value);
            
            // Validation
            if (models.length === 0) {
                alert('Please select at least one model');
                return;
            }
            
            if (tickers.length === 0) {
                alert('Please select at least one ticker');
                return;
            }
            
            const thresholds = Array.from(document.querySelectorAll('.threshold-input'))
                .map(inp => parseFloat(inp.value))
                .filter(v => !isNaN(v) && v >= 0);
            
            const regimeConfigs = [];
            
            if (document.getElementById('regime-none').checked) {
                regimeConfigs.push(null);
            }
            
            // Build VIX combinations (Power Set)
            const vixValues = Array.from(document.querySelectorAll('input[name="regime_vix"]:checked'))
                .map(cb => parseInt(cb.value));
            
            if (vixValues.length > 0) {
                const vixCombos = getCombinations(vixValues);
                vixCombos.forEach(combo => {
                    regimeConfigs.push({col: 'regime_vix', allowed: combo.sort((a,b) => a-b)});
                });
            }
            
            // Build GMM combinations (Power Set)
            const gmmValues = Array.from(document.querySelectorAll('input[name="regime_gmm"]:checked'))
                .map(cb => parseInt(cb.value));
            
            if (gmmValues.length > 0) {
                const gmmCombos = getCombinations(gmmValues);
                gmmCombos.forEach(combo => {
                    regimeConfigs.push({col: 'regime_gmm', allowed: combo.sort((a,b) => a-b)});
                });
            }
            
            const payload = {
                models: models,
                tickers: tickers,
                thresholds: thresholds.length > 0 ? thresholds : [0.0, 0.001, 0.002],
                z_score: [true, false],
                use_bot: [false],
                volatility_normalization: [false],
                regime_configs: regimeConfigs
            };
            
            console.log('Sending batch request:', payload);
            
            try {
                const res = await fetch('/api/create_batch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                
                if (!res.ok) {
                    throw new Error('HTTP ' + res.status + ': ' + res.statusText);
                }
                
                const result = await res.json();
                
                if (result.error) {
                    alert('Error: ' + result.error);
                    return;
                }
                
                alert('Batch Created! ' + result.jobs_created + ' jobs queued. Batch ID: ' + result.batch_id);
                refreshStats();
                
            } catch (error) {
                console.error('Batch creation failed:', error);
                alert('Failed to create batch: ' + error.message);
            }
        }
        
        setInterval(refreshStats, 2000);
        window.onload = refreshStats;
    </script>
</head>
<body>
    <h1>Optimization Command & Control - 3D Strategy Heatmap</h1>
    
    <div class="card">
        <h2>Job Queue Status</h2>
        <div style="display: flex; gap: 10px;">
            <div class="stat-box">
                <div class="stat-label">Pending</div>
                <div id="s-pending" class="stat-val">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Running</div>
                <div id="s-running" class="stat-val">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Completed</div>
                <div id="s-completed" class="stat-val">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Failed</div>
                <div id="s-failed" class="stat-val">0</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Active Workers</h2>
        <div id="workers-list"></div>
    </div>

    <div class="card">
        <h2>Launch 3D Grid Search</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <label>Models (Axis X)</label>
                <select id="models" multiple style="height: 120px;">
                    __MODEL_OPTIONS__
                </select>
            </div>
            <div>
                <label>Tickers (Test Assets)</label>
                <select id="tickers" multiple style="height: 120px;">
                    __TICKER_OPTIONS__
                </select>
            </div>
        </div>
        
        <div class="grid-config">
            <div class="config-section">
                <h3>Thresholds (Axis Z)</h3>
                <label>Min Confidence Levels:</label>
                <input type="number" class="threshold-input" value="0.0001" step="0.0001" min="0">
                <input type="number" class="threshold-input" value="0.0005" step="0.0001" min="0">
                <input type="number" class="threshold-input" value="0.0010" step="0.0001" min="0">
                <input type="number" class="threshold-input" value="0.0020" step="0.0001" min="0">
            </div>
            
            <div class="config-section">
                <h3>Regime Filters (Axis Y)</h3>
                <p style="font-size: 11px; color: #b0bec5; margin-bottom: 10px;">
                    Simulations created for all combinations (subsets) of checked regimes.
                </p>
                
                <label style="display: block; margin-bottom: 8px;">
                    <input type="checkbox" id="regime-none" checked> 
                    <strong>No Filter (Baseline)</strong>
                </label>
                
                <hr style="border: none; border-top: 1px solid #3a4a6a; margin: 12px 0;">
                
                <label style="display: block; font-weight: 600; margin-bottom: 6px;">
                    VIX Regime:
                </label>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 12px;">
                    <label><input type="checkbox" name="regime_vix" value="0" checked> 0 - Bear Volatile</label>
                    <label><input type="checkbox" name="regime_vix" value="1" checked> 1 - Bear Quiet</label>
                    <label><input type="checkbox" name="regime_vix" value="2" checked> 2 - Bull Volatile</label>
                    <label><input type="checkbox" name="regime_vix" value="3" checked> 3 - Bull Quiet</label>
                </div>
                
                <hr style="border: none; border-top: 1px solid #3a4a6a; margin: 12px 0;">
                
                <label style="display: block; font-weight: 600; margin-bottom: 6px;">
                    GMM Regime:
                </label>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                    <label><input type="checkbox" name="regime_gmm" value="0" checked> 0 - Low Vol</label>
                    <label><input type="checkbox" name="regime_gmm" value="1" checked> 1 - Medium Vol</label>
                    <label><input type="checkbox" name="regime_gmm" value="2" checked> 2 - High Vol</label>
                    <label><input type="checkbox" name="regime_gmm" value="3" checked> 3 - Extreme</label>
                </div>
            </div>
            
            <div class="config-section">
                <h3>Advanced Options</h3>
                <label><input type="checkbox" id="z-score"> Z-Score Outlier Removal</label>
                <label><input type="checkbox" id="vol-norm"> Volatility Normalization</label>
                <label><input type="checkbox" id="use-bot"> Enable Trading Bot Filter</label>
            </div>
        </div>
        
        <div class="heatmap-hint">
            <strong>Strategy Heatmap:</strong> This grid search creates a 3D optimization space: Model x Regime x Threshold. 
            Look for "Clusters of Success" where Hit Rate > 55% and trades > 10.
        </div>
        
        <br>
        <button class="btn" onclick="createJob()">Queue Heatmap Batch</button>
    </div>

    <div class="card">
        <h2>Top Strategies (Sorted by SQN)</h2>
        <p style="font-size: 12px; color: #b0bec5; margin-top: -10px;">
            System Quality Number: >3 = Excellent, 2-3 = Good, 1-2 = Fair, <1 = Poor
        </p>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Ticker</th>
                    <th>Model</th>
                    <th>SQN</th>
                    <th>Return %</th>
                    <th>Hit Rate</th>
                    <th>Trades</th>
                    <th>Expectancy</th>
                    <th>Profit Factor</th>
                    <th>Threshold</th>
                    <th>Z-Score</th>
                    <th>Vol Norm</th>
                    <th>Use Bot</th>
                    <th>Regime Col</th>
                    <th>Allowed Regimes</th>
                    <th>Sim Methods</th>
                    <th>Params</th>
                </tr>
            </thead>
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
    
    html = HTML_TEMPLATE.replace("__MODEL_OPTIONS__", m_opts)
    html = html.replace("__TICKER_OPTIONS__", t_opts)
    return html

@app.get("/api/stats")
def get_stats():
    return database.get_dashboard_stats()

@app.post("/api/create_batch")
async def create_batch(config: dict):
    try:
        log.info(f"Received batch creation request: {config}")
        
        jobs = []
        models = config.get("models", [])
        tickers = config.get("tickers", [])
        thresholds = config.get("thresholds", [0.0])
        z_scores = config.get("z_score", [False])
        use_bots = config.get("use_bot", [False])
        vol_norms = config.get("volatility_normalization", [False])
        regime_configs = config.get("regime_configs", [None])
        
        if not models:
            log.error("No models selected")
            return {"error": "Please select at least one model", "batch_id": None, "jobs_created": 0}
        
        if not tickers:
            log.error("No tickers selected")
            return {"error": "Please select at least one ticker", "batch_id": None, "jobs_created": 0}
        
        log.info(f"Building grid: {len(models)} models x {len(tickers)} tickers x {len(thresholds)} thresholds x {len(regime_configs)} regimes")
        
        for m in models:
            for t in tickers:
                for thresh in thresholds:
                    for z in z_scores:
                        for bot in use_bots:
                            for vol in vol_norms:
                                for regime_cfg in regime_configs:
                                    job_params = {
                                        "model_id": m,
                                        "ticker": t,
                                        "initial_cash": 10000,
                                        "use_bot": bot,
                                        "min_prediction_threshold": thresh,
                                        "enable_z_score_check": z,
                                        "volatility_normalization": vol
                                    }
                                    
                                    if regime_cfg and regime_cfg.get("col") and regime_cfg.get("allowed"):
                                        job_params["regime_col"] = regime_cfg["col"]
                                        job_params["allowed_regimes"] = regime_cfg["allowed"]
                                    
                                    jobs.append(job_params)
        
        log.info(f"Created {len(jobs)} job configurations")
        
        batch_id = database.create_jobs(jobs)
        
        log.info(f"✓ Batch {batch_id} created successfully with {len(jobs)} jobs")
        
        return {"batch_id": batch_id, "jobs_created": len(jobs)}
        
    except Exception as e:
        log.error(f"Failed to create batch: {e}", exc_info=True)
        return {"error": str(e), "batch_id": None, "jobs_created": 0}

@app.post("/api/worker/heartbeat")
def worker_heartbeat_endpoint(payload: dict):
    worker_id = payload.get("worker_id")
    job_id = payload.get("job_id")
    database.worker_heartbeat(worker_id, job_id)
    return {"status": "ok"}

@app.post("/api/worker/claim")
def claim_job(payload: dict):
    worker_id = payload.get("worker_id", "unknown")
    job = database.claim_job(worker_id)
    return job

@app.post("/api/worker/complete")
def complete_job(payload: dict):
    job_id = payload.get("job_id")
    result = payload.get("result")
    status = payload.get("status", "COMPLETED")
    database.complete_job(job_id, result, status)
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

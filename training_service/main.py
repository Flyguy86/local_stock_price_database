from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from .config import settings
from .db import db
from .trainer import start_training, train_model_task, ALGORITHMS
from .data import get_data_options

settings.ensure_paths()
logging.basicConfig(level=settings.log_level)
log = logging.getLogger("training.api")

app = FastAPI(title="Training Service")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title> Training Service Dashboard</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        :root {
          --bg: #0f172a;
          --bg-card: #1e293b;
          --text: #e2e8f0;
          --text-muted: #94a3b8;
          --border: #334155;
          --primary: #8b5cf6;
          --primary-hover: #7c3aed;
          --danger: #ef4444;
          --success: #10b981;
        }
        body { margin: 0; font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }
        * { box-sizing: border-box; }
        
        .layout { max-width: 1200px; margin: 0 auto; padding: 2rem; display: grid; gap: 2rem; }
        
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }
        h1 { margin: 0; font-size: 1.5rem; font-weight: 600; background: linear-gradient(to right, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        section { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
        h2 { margin: 0; font-size: 1.1rem; font-weight: 600; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
        
        .row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; }
        .group { display: flex; flex-direction: column; gap: 0.25rem; }
        
        label { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }
        input, select { background: rgba(0,0,0,0.2); border: 1px solid var(--border); color: var(--text); padding: 0.5rem; border-radius: 4px; outline: none;  }
        input:focus, select:focus { border-color: var(--primary); }
        
        button { background: var(--primary); color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        button:hover { background: var(--primary-hover); transform: translateY(-1px); }
        button.secondary { background: var(--border); }
        button.secondary:hover { background: #475569; }
        
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { text-align: left; padding: 0.75rem; border-bottom: 1px solid var(--border); }
        th { color: var(--text-muted); font-weight: 600; font-size: 0.85rem; }
        tr:hover { background: rgba(255,255,255,0.02); }
        
        .badge { padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; background: var(--border); }
        .badge.completed { background: rgba(16, 185, 129, 0.2); color: #34d399; }
        .badge.failed { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
        .badge.running { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
        
        pre { background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 6px; overflow: auto; }
        
        dialog {
            background: var(--bg-card);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            max-width: 600px;
            width: 90%;
        }
        dialog::backdrop {
            background: rgba(0,0,0,0.5);
            backdrop-filter: blur(2px);
        }
      </style>
    </head>
    <body>
      <div class="layout">
        <header>
            <div style="display:flex; align-items:center; gap:1rem;">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2"><path d="M12 2v20"/><path d="M2 12h20"/><path d="M2 2l20 20"/><path d="M22 2 2 22"/></svg>
                <h1>Training Service</h1>
            </div>
            <div class="badge">Port: 8200</div>
        </header>

        <dialog id="report-modal">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:1px solid var(--border); padding-bottom:0.5rem">
                <h2 style="margin:0; border:none; color: var(--primary);">Training Report</h2>
                <button class="secondary" onclick="closeReport()">Close</button>
            </div>
            <div id="report-content" style="max-height: 70vh; overflow-y: auto;"></div>
        </dialog>
        
        <section>
            <h2>Start New Training</h2>
            <div class="row">
                <div class="group">
                    <label>Algorithm</label>
                    <select id="algo"></select>
                </div>
                <div class="group">
                     <label>Symbol</label>
                     <input id="symbol" placeholder="e.g. NVDA" style="text-transform:uppercase;" onchange="loadOptions()">
                </div>
                <div class="group">
                     <label>Data Options</label>
                     <select id="data_options" style="max-width: 200px;"><option value="">All / Default</option></select>
                </div>
                <div class="group">
                     <label>Target</label>
                     <select id="target">
                        <option value="close">Close</option>
                        <option value="open">Open</option>
                        <option value="high">High</option>
                        <option value="low">Low</option>
                        <option value="volume">Volume</option>
                        <option value="vwap">VWAP</option>
                     </select>
                </div>
                <button onclick="train()" style="margin-top:auto">Start Training Job</button>
            </div>
        </section>
        
        <section>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2>Model Registry</h2>
                <button class="secondary" onclick="loadModels()">Refresh</button>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Algo</th>
                        <th>Symbol</th>
                        <th>Status</th>
                        <th>Metrics</th>
                        <th>Created</th>
                    </tr>
                </thead>
                <tbody id="models-body"></tbody>
            </table>
        </section>
      </div>
      
      <script>
        const $ = id => document.getElementById(id);
        const modelsCache = {};
        
        async function load() {
            // Load Algos
            const res = await fetch('/algorithms');
            const algos = await res.json();
            $('algo').innerHTML = algos.map(a => `<option value="${a}">${a}</option>`).join('');
            
            // Load Models
            loadModels();
        }
        
        async function loadOptions() {
             const symbol = $('symbol').value.trim().toUpperCase();
             if(!symbol) {
                 $('data_options').innerHTML = '<option value="">All / Default</option>';
                 return;
             }
             try {
                const res = await fetch('/data/options/' + symbol);
                const opts = await res.json();
                $('data_options').innerHTML = '<option value="">All / Default</option>' + opts.map(o => `<option value='${o}'>${o}</option>`).join('');
             } catch(e) { console.error(e); }
        }

        function showReport(id) {
            const m = modelsCache[id];
            if(!m || !m.metrics) return;
            
            let metrics = {};
            try {
                metrics = JSON.parse(m.metrics);
            } catch(e) {
                $('report-content').innerHTML = `Error parsing metrics: ${m.metrics}`;
                $('report-modal').showModal();
                return;
            }

            let html = `<h3>Performance Metrics</h3><ul>`;
            // Prioritize standard metrics
            const priority = ['mse', 'rmse', 'accuracy', 'features_count'];
            priority.forEach(k => {
                if(metrics[k] !== undefined) html += `<li><b style="color:var(--text-muted)">${k.toUpperCase()}:</b> <span style="font-family:monospace">${metrics[k].toFixed ? metrics[k].toFixed(5) : metrics[k]}</span></li>`;
            });
            html += `</ul>`;

            // Dropped Columns
            if(metrics.dropped_cols && metrics.dropped_cols.length > 0) {
                html += `
                <div style="margin-top:1rem; padding:0.5rem; background:rgba(239, 68, 68, 0.1); border-radius:4px; border:1px solid rgba(239, 68, 68, 0.3)">
                    <h4 style="margin:0 0 0.5rem 0; font-size:0.9rem; color: #fca5a5">⚠️ Pruned Features (All-NaN)</h4>
                    <div style="font-size:0.8rem; font-family:monospace; color:var(--text-muted)">${metrics.dropped_cols.join(', ')}</div>
                </div>`;
            }

            // Feature Importance
            if(metrics.feature_details) {
                html += `<h3>Feature Analysis</h3>
                <div style="overflow-x:auto;">
                <table style="width:100%; font-size:0.9rem; border-collapse: collapse;">
                    <thead>
                        <tr style="background:rgba(255,255,255,0.05)">
                            <th style="padding:0.5rem; text-align:left">Feature</th>
                            <th style="padding:0.5rem; text-align:right" title="Mean Absolute SHAP Value">SHAP</th>
                            <th style="padding:0.5rem; text-align:right" title="Permutation Importance">Permutation</th>
                            <th style="padding:0.5rem; text-align:right" title="Coefficient or Tree Importance">Coeff / Imp</th>
                        </tr>
                    </thead>
                    <tbody>`;
                
                const rows = Object.entries(metrics.feature_details).map(([feat, det]) => {
                    // determine sort key (shap > perm > coeff)
                    const shap = det.shap_mean_abs || 0;
                    const perm = det.permutation_mean || 0;
                    const coeff = det.coefficient !== undefined ? det.coefficient : (det.tree_importance || 0);
                    return {feat, shap, perm, coeff};
                });
                
                // Sort by SHAP desc, then Permutation
                rows.sort((a,b) => Math.abs(b.shap) - Math.abs(a.shap) || Math.abs(b.perm) - Math.abs(a.perm));
                
                rows.slice(0, 50).forEach(r => {
                    const fmt = n => Math.abs(n) < 0.0001 && n !== 0 ? n.toExponential(2) : n.toFixed(5);
                    html += `
                    <tr style="border-bottom:1px solid #334155">
                        <td style="padding:0.4rem; font-weight:500">${r.feat}</td>
                        <td style="padding:0.4rem; text-align:right; font-family:monospace; color:${r.shap > 0 ? '#f472b6' : '#94a3b8'}">${fmt(r.shap)}</td>
                        <td style="padding:0.4rem; text-align:right; font-family:monospace; color:${r.perm > 0 ? '#4ade80' : '#94a3b8'}">${fmt(r.perm)}</td>
                        <td style="padding:0.4rem; text-align:right; font-family:monospace; color:${Math.abs(r.coeff) > 0 ? '#60a5fa' : '#94a3b8'}">${fmt(r.coeff)}</td>
                    </tr>`;
                });
                 html += `</tbody></table></div>`;
            } else if(metrics.feature_importance) {
                html += `<h3>Feature Importance</h3>
                <table style="width:100%; font-size:0.9rem">
                    <thead><tr><th style="padding:0.25rem">Feature</th><th style="padding:0.25rem; text-align:right">Score</th></tr></thead>
                    <tbody>`;
                
                // Convert dict to array and take top 20
                const sorted = Object.entries(metrics.feature_importance); // Already sorted in backend but good to be safe
                
                sorted.slice(0, 50).forEach(([feat, score]) => {
                    html += `<tr><td style="padding:0.25rem; border-bottom:1px solid #334155">${feat}</td><td style="padding:0.25rem; text-align:right; font-family:monospace; border-bottom:1px solid #334155">${score.toFixed(5)}</td></tr>`;
                });
                
                html += `</tbody></table>`;
                if(sorted.length > 50) html += `<div style="text-align:center; font-size:0.8rem; padding:0.5rem; color:var(--text-muted)">... ${sorted.length - 50} more features ...</div>`
            }

            $('report-content').innerHTML = html;
            $('report-modal').showModal();
        }

        function closeReport() {
            $('report-modal').close();
        }
        
        async function loadModels() {
            // Keep track of open errors
            const openErrors = new Set();
            document.querySelectorAll('details').forEach(el => {
                if(el.open && el.id) openErrors.add(el.id);
            });

            const res = await fetch('/models');
            const models = await res.json();
            const tbody = $('models-body');
            
            if(!models.length) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color: var(--text-muted)">No models found</td></tr>';
                return;
            }
            
            tbody.innerHTML = models.map(m => {
                modelsCache[m.id] = m; // Update cache

                let statusHtml = `<span class="badge ${m.status}">${m.status}</span>`;
                if(m.status === 'failed' && m.error_message) {
                    const detailId = 'err-' + m.id;
                    const isOpen = openErrors.has(detailId) ? 'open' : '';
                    statusHtml += `
                    <details id="${detailId}" ${isOpen} style="margin-top:0.5rem; color: var(--danger); font-size: 0.85em;">
                        <summary style="cursor:pointer; opacity: 0.8;">Show Error</summary>
                        <div style="background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 4px; margin-top: 0.25rem; white-space: pre-wrap; font-family: monospace;">${m.error_message}</div>
                    </details>`;
                }
                
                let metricsBtn = '<span style="color:var(--text-muted)">-</span>';
                if(m.status === 'completed' && m.metrics && m.metrics !== '{}') {
                     metricsBtn = `<button class="secondary" style="font-size:0.75rem; padding:0.25rem 0.5rem;" onclick="showReport('${m.id}')">View Report</button>`;
                }

                return `
                <tr style="vertical-align: top;">
                    <td><span class="badge" title="${m.id}">${m.id.substring(0,8)}</span></td>
                    <td>${m.name}</td>
                    <td>${m.algorithm}</td>
                    <td>${m.symbol}</td>
                    <td>${statusHtml}</td>
                    <td>${metricsBtn}</td>
                    <td>${m.created_at.replace('T', ' ').split('.')[0]}</td>
                </tr>
            `}).join('');
        }
        
        async function train() {
            const symbol = $('symbol').value.trim().toUpperCase();
            if(!symbol) return alert('Symbol required');
            
            const btn = document.querySelector('button');
            btn.disabled = true;
            btn.innerText = 'Starting...';
            
            try {
                const res = await fetch('/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbol,
                        algorithm: $('algo').value,
                        target_col: $('target').value,
                        data_options: $('data_options').value || null
                    })
                });
                
                if(!res.ok) {
                    const err = await res.json();
                    alert('Error: ' + err.detail);
                } else {
                    loadModels();
                }
            } catch(e) {
                alert('Request failed: ' + e);
            } finally {
                btn.disabled = false;
                btn.innerText = 'Start Training Job';
            }
        }
        
        window.onload = load;
        setInterval(loadModels, 5000);
      </script>
    </body>
    </html>
    """

@app.get("/algorithms")
def get_algorithms():
    return list(ALGORITHMS.keys())

@app.get("/data/options/{symbol}")
def list_options(symbol: str):
    return get_data_options(symbol)

class TrainRequest(BaseModel):
    symbol: str
    algorithm: str
    target_col: str = "close"
    hyperparameters: Optional[Dict[str, Any]] = None
    data_options: Optional[str] = None

@app.get("/algorithms")
def list_algorithms():
    return list(ALGORITHMS.keys())

@app.get("/models")
def list_models():
    return db.list_models()

@app.get("/models/{model_id}")
def get_model(model_id: str):
    model = db.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    # Convert result tuples/rows to dict if necessary (DuckDB fetchone returns tuple)
    # The get_model in db.py returns a tuple, let's map it. 
    # Actually, simpler to return as part of list or fix db.py to return dict.
    # For now, let's rely on list_models for UI.
    return {"id": model_id, "data": str(model)}

@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithm must be one of {list(ALGORITHMS.keys())}")
    
    training_id = start_training(req.symbol, req.algorithm, req.target_col, req.hyperparameters, req.data_options)
    
    background_tasks.add_task(
        train_model_task, 
        training_id, 
        req.symbol, 
        req.algorithm, 
        req.target_col, 
        req.hyperparameters or {},
        req.data_options
    )
    
    return {"id": training_id, "status": "started"}

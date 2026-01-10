from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from pathlib import Path

from .config import settings
from .db import db
from .trainer import start_training, train_model_task, ALGORITHMS
from .data import get_data_options, get_feature_map

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
                     <label>Data Options</label>
                     <select id="data_options" style="max-width: 300px;" onchange="updateSymbols()"><option value="">Loading...</option></select>
                </div>
                <div class="group">
                     <label>Symbol (Target)</label>
                     <select id="symbol" style="min-width: 120px;"><option value="">Select Config First</option></select>
                </div>
                <div class="group">
                     <label>Context 1</label>
                     <select id="ctx1" class="ctx-select" style="min-width: 100px;"><option value="">(None)</option></select>
                </div>
                <div class="group">
                     <label>Context 2</label>
                     <select id="ctx2" class="ctx-select" style="min-width: 100px;"><option value="">(None)</option></select>
                </div>
                <div class="group">
                     <label>Context 3</label>
                     <select id="ctx3" class="ctx-select" style="min-width: 100px;"><option value="">(None)</option></select>
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
                <div class="group">
                     <label>Timeframe</label>
                     <select id="timeframe">
                        <option value="1m">1 min</option>
                        <option value="10m">10 min</option>
                        <option value="30m">30 min</option>
                        <option value="1h">1 hour</option>
                        <option value="4h">4 hours</option>
                        <option value="8h">8 hours</option>
                     </select>
                </div>
                <div class="group">
                     <label>Parent Model (Features)</label>
                     <select id="parent_model" style="max-width:200px" onchange="onParentModelChange()"><option value="">(None)</option></select>
                </div>
                <button onclick="train()" style="margin-top:auto">Start Training Job</button>
            </div>
            
             <!-- Feature Selection UI (Hidden by default) -->
            <div id="feature-selection-ui" style="display:none; margin-top: 1.5rem; border-top: 1px solid var(--border); padding-top: 1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                     <h3 style="font-size:1rem; margin:0">Parent Feature Selection</h3>
                     <div style="font-size:0.8rem; color:var(--text-muted)">
                        Uncheck to prune features from the new model
                     </div>
                </div>
                
                <div style="background: rgba(59, 130, 246, 0.1); padding: 0.75rem; border-radius: 6px; font-size: 0.85rem; margin-bottom: 1rem; border: 1px solid rgba(59, 130, 246, 0.2);">
                    <strong style="color: #93c5fd;">Metric Guide:</strong>
                    <ul style="margin: 0.5rem 0 0 1.2rem; padding: 0; display: grid; gap: 0.5rem; grid-template-columns: 1fr 1fr;">
                        <li>
                            <strong style="color: #f472b6;">SHAP (Mean Abs)</strong>: Impact magnitude. Higher is better. <br>
                            <span style="opacity:0.7">&lt; 0.001 is often noise. &gt; 0.1 is strong.</span>
                        </li>
                        <li>
                            <strong style="color: #4ade80;">Permutation</strong>: Importance via randomization. <br>
                            <span style="opacity:0.7">&gt; 0 is good. &lt;= 0 is useless/harmful.</span>
                        </li>
                        <li>
                            <strong style="color: #60a5fa;">Coeff / Imp</strong>: Linear weight or Tree split %. <br>
                            <span style="opacity:0.7">Large magnitude (>10) may indicate instability.</span>
                        </li>
                    </ul>
                </div>
                
                <div style="max-height: 400px; overflow-y: auto; border: 1px solid var(--border); border-radius: 4px;">
                    <table style="width:100%; font-size:0.85rem; border-collapse: collapse;">
                        <thead style="position: sticky; top: 0; background: var(--bg-card); z-index: 1;">
                            <tr>
                                <th style="width: 30px;"><input type="checkbox" checked onclick="toggleAllFeatures(this)"></th>
                                <th style="text-align:left;">
                                    Feature <br>
                                    <input type="text" id="feat-filter" placeholder="Contains..." style="width: 100%; font-size: 0.75rem; padding: 2px; margin-top: 2px; background: rgba(0,0,0,0.3); border: 1px solid var(--border);" onkeyup="filterFeatures()">
                                </th>
                                <th style="text-align:right;">SHAP</th>
                                <th style="text-align:right;">Permutation</th>
                                <th style="text-align:right;">Coeff / Imp</th>
                            </tr>
                        </thead>
                        <tbody id="parent-features-body"></tbody>
                    </table>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-muted); text-align: right;">
                    Selected: <span id="selected-count">0</span> / <span id="total-count">0</span>
                </div>
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
                        <th>TF</th>
                        <th>Status</th>
                        <th>Metrics</th>
                        <th>Created</th>
                        <th>Actions</th>
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
            
            // Load Data Options (Global)
            loadOptions();

            // Load Models
            loadModels();
        }
        
        async function onParentModelChange() {
            const pid = $('parent_model').value;
            const ui = $('feature-selection-ui');
            const tbody = $('parent-features-body');
            
            if(!pid) {
                ui.style.display = 'none';
                return;
            }
            
            const m = modelsCache[pid];
            if(!m) return;
            
            // 1. Pre-fill Config
            if(m.algorithm) $('algo').value = m.algorithm;
            if(m.target_col) $('target').value = m.target_col;
            if(m.timeframe) $('timeframe').value = m.timeframe;
            if(m.data_options) {
                $('data_options').value = m.data_options;
                updateSymbols(); // Trigger updates
                // Wait a tick for symbols to populate? Usually sync enough 
                // if(m.symbol) $('symbol').value = m.symbol.split(',')[0];
                // Handling complex symbol string "NVDA,SPY,QQQ"
                if(m.symbol) {
                     const parts = m.symbol.split(',');
                     $('symbol').value = parts[0]; 
                     // Try to fill contexts
                     const ctxs = document.querySelectorAll('.ctx-select');
                     for(let i=0; i<3; i++) {
                         if(parts[i+1] && ctxs[i]) ctxs[i].value = parts[i+1];
                         else if(ctxs[i]) ctxs[i].value = "";
                     }
                }
            }

            // 2. Render Feature Table
            let metrics = {};
            try { metrics = JSON.parse(m.metrics); } catch(e){}
            
            let rows = [];
            
            if(metrics.feature_details) {
                rows = Object.entries(metrics.feature_details).map(([feat, det]) => {
                    const shap = det.shap_mean_abs || 0;
                    const perm = det.permutation_mean || 0;
                    const coeff = det.coefficient !== undefined ? det.coefficient : (det.tree_importance || 0);
                    return {feat, shap, perm, coeff};
                });
                // Sort
                rows.sort((a,b) => Math.abs(b.shap) - Math.abs(a.shap));
            } else if (metrics.feature_importance) {
                // Fallback for older models
                rows = Object.entries(metrics.feature_importance).map(([feat, score]) => ({
                    feat, shap: 0, perm: 0, coeff: score // treat simple score as coeff for display
                }));
            } else {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding:1rem">No feature details available in parent</td></tr>';
                ui.style.display = 'block';
                return;
            }
            
            const fmt = n => Math.abs(n) < 0.0001 && n !== 0 ? n.toExponential(2) : n.toFixed(5);
            
            tbody.innerHTML = rows.map(r => `
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="text-align:center;">
                        <input type="checkbox" class="feat-check" value="${r.feat}" checked onchange="updateCount()">
                    </td>
                    <td style="padding:0.4rem; font-family:monospace; font-size:0.8rem">${r.feat}</td>
                    <td style="padding:0.4rem; text-align:right; font-family:monospace; color:${r.shap > 0 ? '#f472b6' : '#94a3b8'}">${fmt(r.shap)}</td>
                    <td style="padding:0.4rem; text-align:right; font-family:monospace; color:${r.perm > 0 ? '#4ade80' : '#94a3b8'}">${fmt(r.perm)}</td>
                    <td style="padding:0.4rem; text-align:right; font-family:monospace; color:${Math.abs(r.coeff) > 0 ? '#60a5fa' : '#94a3b8'}">${fmt(r.coeff)}</td>
                </tr>
            `).join('');
            
            $('total-count').innerText = rows.length;
            updateCount();
            
            ui.style.display = 'block';
        }

        function filterFeatures() {
            const term = $('feat-filter').value.toLowerCase();
            const rows = document.querySelectorAll('#parent-features-body tr');
            rows.forEach(r => {
                const feat = r.querySelector('td:nth-child(2)').innerText.toLowerCase();
                r.style.display = feat.includes(term) ? '' : 'none';
            });
        }

        function toggleAllFeatures(el) {
            const visibleOnly = true; // should toggle only visible? typically yes in filters
            const rows = document.querySelectorAll('#parent-features-body tr');
            rows.forEach(r => {
                if(r.style.display !== 'none') {
                    const cb = r.querySelector('.feat-check');
                    if(cb) cb.checked = el.checked;
                }
            });
            updateCount();
        }

        function updateCount() {
            const count = document.querySelectorAll('.feat-check:checked').length;
            $('selected-count').innerText = count;
        }

        let featureMap = {};

        async function loadOptions() {
             try {
                const res = await fetch('/data/map');
                featureMap = await res.json();
                
                const select = $('data_options');
                const options = Object.keys(featureMap).sort();
                
                if (options.length === 0) {
                     select.innerHTML = '<option value="">No features found</option>';
                     return;
                }
                
                select.innerHTML = '<option value="">-- Select Configuration --</option>' + options.map(o => {
                    let label = o;
                    try {
                        const j = JSON.parse(o);
                        if (j.train_window && j.test_window) {
                            label = `Train:${j.train_window} Test:${j.test_window}`; 
                        }
                    } catch(e) {}
                    if(label.length > 60) label = label.substring(0,57) + '...';
                    return `<option value='${o}'>${label}</option>`;
                }).join('');
                
                updateSymbols();
             } catch(e) { console.error(e); }
        }

        function updateSymbols() {
            const opt = $('data_options').value;
            const symSelect = $('symbol');
            const ctxSelects = document.querySelectorAll('.ctx-select');
            
            symSelect.disabled = false;
            
            if(!opt) {
                symSelect.innerHTML = '<option value="">Select Config First</option>';
                symSelect.disabled = true;
                ctxSelects.forEach(s => {
                    s.innerHTML = '<option value="">(None)</option>';
                    s.disabled = true;
                });
                return;
            }
            
            const symbols = featureMap[opt] || [];
            let html = '';
            
            if (symbols.length === 0) {
                html = '<option value="">No symbols found</option>';
            } else {
                html = symbols.map(s => `<option value="${s}">${s}</option>`).join('');
            }
            
            symSelect.innerHTML = html;
            
            // Populate context dropdowns
            ctxSelects.forEach(s => {
                s.disabled = false;
                s.innerHTML = '<option value="">(None)</option>' + html;
            });
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
            
            const definitions = {
                'mse': { title: 'Mean Squared Error', desc: 'Average squared difference between predicted and actual values. Lower is better. Punishes large errors heavily.' },
                'rmse': { title: 'Root Mean Squared Error', desc: 'Square root of MSE. Same unit as the target (e.g. price $). Lower is better. Good/Bad depends on stock price (e.g. RMSE 2.0 is great for NVDA @ 1000, bad for penny stocks).' },
                'accuracy': { title: 'Accuracy Score', desc: 'Percentage of correct predictions. 0.5 is random guessing for binary. >0.55 is often considered "good" in trading. >0.65 is suspicious (overfitting?).' },
                'features_count': { title: 'Features Used', desc: 'Number of input variables used by the model.' }
            };

            // Prioritize standard metrics
            const priority = ['mse', 'rmse', 'accuracy', 'features_count'];
            priority.forEach(k => {
                if(metrics[k] !== undefined) {
                    const def = definitions[k] || { title: k.toUpperCase(), desc: '' };
                    html += `
                    <li style="margin-bottom: 0.5rem">
                        <div style="display:flex; justify-content:space-between; align-items:baseline">
                            <b style="color:var(--text-muted)">${def.title}</b>
                            <span style="font-family:monospace; font-size:1.1em; color:var(--text)">${metrics[k].toFixed ? metrics[k].toFixed(5) : metrics[k]}</span>
                        </div>
                        <div style="font-size:0.8rem; color:var(--text-muted); opacity:0.8; margin-top:2px;">${def.desc}</div>
                    </li>`;
                }
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
        
        async function deleteModel(id) {
            if(!confirm('Are you sure you want to delete this model?')) return;
            try {
                const res = await fetch('/models/' + id, { method: 'DELETE' });
                if(res.ok) {
                    loadModels();
                } else {
                    alert('Failed to delete');
                }
            } catch(e) { console.error(e); }
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
            
            // Populate Parent Dropdown
            // Only completed models can be parents? Or any? Let's say any except failed.
            const parentSel = $('parent_model');
            const currentParent = parentSel.value;
            parentSel.innerHTML = '<option value="">(None)</option>' + 
                models
                .filter(m => m.status === 'completed')
                .map(m => `<option value="${m.id}">${m.name} (${m.algorithm})</option>`)
                .join('');
            parentSel.value = currentParent; // Restore selection

            if(!models.length) {
                tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; color: var(--text-muted)">No models found</td></tr>';
                return;
            }
            
            // Reconstruct Tree
            const map = {};
            const roots = [];
            models.forEach(m => {
                m.children = [];
                map[m.id] = m;
            });
            
            // Link Sort (descending time usually, but for tree maybe strictly by parent?)
            // We want roots (null parent) first.
            models.forEach(m => {
                if(m.parent_model_id && map[m.parent_model_id]) {
                    map[m.parent_model_id].children.push(m);
                } else {
                    roots.push(m);
                }
            });
            
            // Recursive Render
            let html = '';
            
            const renderNode = (m, depth) => {
                modelsCache[m.id] = m;
                
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
                
                // Indentation
                const indent = depth * 20;
                const treeIcon = depth > 0 ? `<span style="color:var(--text-muted); margin-right:5px;">↳</span>` : '';

                html += `
                <tr style="vertical-align: top; background: rgba(255,255,255, ${depth * 0.02})">
                    <td style="padding-left: ${10 + indent}px;">
                        ${treeIcon}
                        <span class="badge" title="${m.id}">${m.id.substring(0,8)}</span>
                    </td>
                    <td>${m.name}</td>
                    <td>${m.algorithm}</td>
                    <td>${m.symbol}</td>
                    <td><span style="font-family:monospace; font-size: 0.8em; background: #334155; padding: 2px 4px; border-radius: 3px;">${m.timeframe || '1m'}</span></td>
                    <td>${statusHtml}</td>
                    <td>${metricsBtn}</td>
                    <td>${m.created_at.replace('T', ' ').split('.')[0]}</td>
                    <td style="display:flex; gap:0.5rem">
                        <button class="secondary" style="font-size:0.75rem; padding:0.25rem 0.5rem; color:#60a5fa; border-color:#1e40af" onclick="retrainModel('${m.id}')">Retrain</button>
                        <button class="secondary" style="font-size:0.75rem; padding:0.25rem 0.5rem; color:#fca5a5; border-color:#7f1d1d" onclick="deleteModel('${m.id}')">Delete</button>
                    </td>
                </tr>`;
                
                // Render children
                // Sort children by time desc
                m.children.sort((a,b) => b.created_at.localeCompare(a.created_at));
                m.children.forEach(c => renderNode(c, depth + 1));
            };
            
            // Sort roots by created_at desc
            roots.sort((a,b) => b.created_at.localeCompare(a.created_at));
            roots.forEach(r => renderNode(r, 0));
            
            tbody.innerHTML = html;
        }
        
        async function retrainModel(id) {
            if(!confirm('Start a new training job with the same parameters?')) return;
            const btn = document.querySelector(`button[onclick="retrainModel('${id}')"]`);
            if(btn) btn.innerText = '...';
            
            try {
                const res = await fetch('/retrain/' + id, { method: 'POST' });
                if(res.ok) {
                    loadModels();
                } else {
                    const err = await res.json();
                    alert('Failed to retrain: ' + err.detail);
                }
            } catch(e) { console.error(e); alert('Error: ' + e); }
        }
        
        async function train() {
            const symbol = $('symbol').value.trim().toUpperCase();
            if(!symbol) return alert('Symbol required');
            
            // Collect context
            const ctx = [
                $('ctx1').value,
                $('ctx2').value,
                $('ctx3').value
            ].filter(s => s && s !== symbol); // Remove empty or duplicate of target
            
            // Join with comma
            const fullSymbolString = [symbol, ...ctx].join(',');

            // Gather feature whitelist if parent is selected
            let featureWhitelist = null;
            if($('parent_model').value) {
                const checked = Array.from(document.querySelectorAll('.feat-check:checked')).map(c => c.value);
                // If NO features are checked, maybe warn? Or allow empty (which train task will error on)
                featureWhitelist = checked;
            }

            const btn = document.querySelector('button');
            btn.disabled = true;
            btn.innerText = 'Starting...';
            
            try {
                const res = await fetch('/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbol: fullSymbolString,
                        algorithm: $('algo').value,
                        target_col: $('target').value,
                        data_options: $('data_options').value || null,
                        timeframe: $('timeframe').value,
                        parent_model_id: $('parent_model').value || null,
                        feature_whitelist: featureWhitelist
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

@app.get("/data/options")
def list_global_options():
    return get_data_options()

@app.get("/data/options/{symbol}")
def list_options(symbol: str):
    return get_data_options(symbol)

@app.get("/data/map")
def get_map():
    return get_feature_map()

class TrainRequest(BaseModel):
    symbol: str
    algorithm: str
    target_col: str = "close"
    hyperparameters: Optional[Dict[str, Any]] = None
    data_options: Optional[str] = None
    timeframe: str = "1m"
    p_value_threshold: float = 0.05
    parent_model_id: Optional[str] = None
    feature_whitelist: Optional[list[str]] = None

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

@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    # 1. Get model to find artifact path (optional, or just construct it)
    # We can query DB or just assume standard path if we want, but better to check DB if we stored custom path.
    # However, trainer uses consistent naming: {id}.joblib
    
    # Delete from DB
    db.delete_model(model_id)
    
    # Delete file
    try:
        model_path = settings.models_dir / f"{model_id}.joblib"
        if model_path.exists():
            model_path.unlink()
    except Exception as e:
        log.error(f"Failed to delete model file {model_id}: {e}")
        # We don't fail the request if file is already gone, but good to log it.
        
    return {"status": "deleted", "id": model_id}

@app.post("/retrain/{model_id}")
async def retrain_model(model_id: str, background_tasks: BackgroundTasks):
    import json
    conn = db.get_connection()
    try:
        # Fetch original parameters
        row = conn.execute("SELECT symbol, algorithm, target_col, hyperparameters, data_options, timeframe FROM models WHERE id = ?", [model_id]).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Original model not found")
        
        symbol, algo, target, params_json, d_opt, tf = row
        
        # Parse params
        params = {}
        if params_json:
            try:
                params = json.loads(params_json)
            except:
                log.warning(f"Could not parse hyperparameters for {model_id}, using empty dict")
                
        # Start new training job
        training_id = start_training(symbol, algo, target, params, d_opt, tf)
    
        background_tasks.add_task(
            train_model_task, 
            training_id, 
            symbol, 
            algo, 
            target, 
            params,
            d_opt,
            tf
        )
        return {"id": training_id, "status": "started", "retrained_from": model_id}
            
    except Exception as e:
        log.error(f"Retrain failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithm must be one of {list(ALGORITHMS.keys())}")
    
    # Inject p_value_threshold into parameters for persistence and task usage
    params = req.hyperparameters or {}
    params["p_value_threshold"] = req.p_value_threshold
    
    training_id = start_training(req.symbol, req.algorithm, req.target_col, params, req.data_options, req.timeframe, req.parent_model_id)
    
    background_tasks.add_task(
        train_model_task, 
        training_id, 
        req.symbol, 
        req.algorithm, 
        req.target_col, 
        params,
        req.data_options,
        req.timeframe,
        req.parent_model_id,
        req.feature_whitelist
    )
    
    return {"id": training_id, "status": "started"}

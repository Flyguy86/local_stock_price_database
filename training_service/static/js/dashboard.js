const $ = id => document.getElementById(id);
const modelsCache = {};
const SECTIONS = ['enet', 'xgb', 'lgbm'];
let featureMap = {};

/**
 * Get grid search display info for a model.
 * Shows if model is a grid parent (with child count) or grid child (with alpha/l1_ratio).
 */
function getGridInfo(model, allModels) {
    // Check if this is a grid parent (has children)
    const childCount = model.grid_children_count || 0;
    if (childCount > 0) {
        return `<span class="badge grid-parent grid-indicator" style="cursor:pointer" onclick="showGridDetails('${model.id}')" title="Click to see grid search details">🔍 <span class="count">${childCount}</span> children ✓</span>`;
    }
    
    // Check if this is a grid child
    if (model.is_grid_member && model.parent_model_id) {
        // Try to get hyperparameters to show alpha/l1_ratio
        let paramStr = '';
        try {
            let hp = model.hyperparameters;
            if (typeof hp === 'string') hp = JSON.parse(hp);
            if (hp) {
                const parts = [];
                if (hp.alpha !== undefined) parts.push(`α=${hp.alpha}`);
                if (hp.l1_ratio !== undefined) parts.push(`L1=${hp.l1_ratio}`);
                paramStr = parts.join(' ');
            }
        } catch(e) {}
        
        return `<span class="badge grid-child" style="cursor:pointer" onclick="showGridDetails('${model.parent_model_id}')" title="Click to see parent grid">👶 ${paramStr || 'child'}</span>`;
    }
    
    return '-';
}

/**
 * Show grid search details in a modal.
 * Displays parent model info and all child models in a table.
 */
async function showGridDetails(parentId) {
    const parent = modelsCache[parentId];
    if (!parent) {
        alert('Parent model not found in cache');
        return;
    }
    
    // Find all children of this parent
    const children = Object.values(modelsCache).filter(m => m.parent_model_id === parentId);
    children.sort((a, b) => {
        // Sort by hyperparameters (alpha, l1_ratio)
        try {
            let hpA = typeof a.hyperparameters === 'string' ? JSON.parse(a.hyperparameters) : a.hyperparameters;
            let hpB = typeof b.hyperparameters === 'string' ? JSON.parse(b.hyperparameters) : b.hyperparameters;
            if (!hpA || !hpB) return 0;
            if (hpA.alpha !== hpB.alpha) return (hpA.alpha || 0) - (hpB.alpha || 0);
            return (hpA.l1_ratio || 0) - (hpB.l1_ratio || 0);
        } catch(e) { return 0; }
    });
    
    // Build the modal content
    let html = `
        <div style="margin-bottom:1rem; padding:0.75rem; background:rgba(139,92,246,0.1); border-radius:6px; border:1px solid rgba(139,92,246,0.3)">
            <h3 style="margin:0 0 0.5rem 0; color:#a78bfa">📊 Parent Model</h3>
            <div style="display:grid; grid-template-columns:auto 1fr; gap:0.25rem 1rem; font-size:0.85rem">
                <span style="color:var(--text-muted)">ID:</span><span>${parent.id}</span>
                <span style="color:var(--text-muted)">Name:</span><span>${parent.name}</span>
                <span style="color:var(--text-muted)">Symbol:</span><span>${parent.symbol}</span>
                <span style="color:var(--text-muted)">Algorithm:</span><span>${parent.algorithm}</span>
                <span style="color:var(--text-muted)">Status:</span><span class="badge ${parent.status}">${parent.status}</span>
            </div>
        </div>
    `;
    
    // Grid search verification status
    const completedChildren = children.filter(c => c.status === 'completed').length;
    const failedChildren = children.filter(c => c.status === 'failed').length;
    const totalChildren = children.length;
    
    const isSuccess = parent.status === 'completed' && completedChildren === totalChildren && totalChildren > 0;
    const hasIssues = failedChildren > 0 || parent.status === 'failed';
    
    html += `
        <div style="margin-bottom:1rem; padding:0.75rem; background:${isSuccess ? 'rgba(16,185,129,0.1)' : hasIssues ? 'rgba(239,68,68,0.1)' : 'rgba(59,130,246,0.1)'}; border-radius:6px; border:1px solid ${isSuccess ? 'rgba(16,185,129,0.3)' : hasIssues ? 'rgba(239,68,68,0.3)' : 'rgba(59,130,246,0.3)'}">
            <h3 style="margin:0 0 0.5rem 0; color:${isSuccess ? '#34d399' : hasIssues ? '#fca5a5' : '#60a5fa'}">${isSuccess ? '✅ Grid Search Verified' : hasIssues ? '❌ Grid Search Has Issues' : '⏳ Grid Search In Progress'}</h3>
            <div style="font-size:0.85rem">
                <span style="color:var(--text-muted)">Total Children:</span> <strong>${totalChildren}</strong><br>
                <span style="color:var(--text-muted)">Completed:</span> <strong style="color:#34d399">${completedChildren}</strong><br>
                ${failedChildren > 0 ? `<span style="color:var(--text-muted)">Failed:</span> <strong style="color:#fca5a5">${failedChildren}</strong><br>` : ''}
            </div>
        </div>
    `;
    
    // Children table
    if (children.length > 0) {
        html += `
            <h3 style="margin:0 0 0.5rem 0">👶 Child Models (${children.length})</h3>
            <div style="max-height:300px; overflow-y:auto">
                <table style="width:100%; font-size:0.8rem">
                    <thead>
                        <tr>
                            <th>Alpha (α)</th>
                            <th>L1 Ratio</th>
                            <th>Status</th>
                            <th>R² Score</th>
                            <th>ID</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        for (const child of children) {
            let hp = {};
            try { hp = typeof child.hyperparameters === 'string' ? JSON.parse(child.hyperparameters) : (child.hyperparameters || {}); } catch(e){}
            
            let r2 = '-';
            try {
                let metrics = typeof child.metrics === 'string' ? JSON.parse(child.metrics) : child.metrics;
                if (metrics) {
                    r2 = metrics.r2 || metrics.R2 || metrics.test_r2 || metrics.cv_mean_r2 || '-';
                    if (typeof r2 === 'number') r2 = r2.toFixed(4);
                }
            } catch(e){}
            
            const statusClass = child.status === 'completed' ? 'completed' : child.status === 'failed' ? 'failed' : 'running';
            
            html += `
                <tr>
                    <td>${hp.alpha !== undefined ? hp.alpha : '-'}</td>
                    <td>${hp.l1_ratio !== undefined ? hp.l1_ratio : '-'}</td>
                    <td><span class="badge ${statusClass}">${child.status}</span></td>
                    <td>${r2}</td>
                    <td><span class="badge" title="${child.id}">${child.id.substring(0,8)}</span></td>
                </tr>
            `;
        }
        
        html += `</tbody></table></div>`;
    } else {
        html += `<p style="color:var(--text-muted)">No child models found yet. Grid search may still be running.</p>`;
    }
    
    $('grid-title').innerText = `🔍 Grid Search: ${parent.name}`;
    $('grid-content').innerHTML = html;
    $('grid-modal').showModal();
}

function closeGridModal() { $('grid-modal').close(); }

function load() {
    console.log("Dashboard load() started");
    loadOptions();
    loadModels();
    updateLogs();
    
    setInterval(updateLogs, 3000);
    setInterval(loadModels, 5000);
}

// --- LOGS ---
async function updateLogs() {
    try {
        const res = await fetch('/logs');
        if(res.ok) {
            const logs = await res.json();
            const div = $('log-container');
            if(div) {
                 const wasAtBottom = div.scrollTop + div.clientHeight >= div.scrollHeight - 20;
                 div.innerText = logs.length ? logs.join('\n') : "No logs yet.";
                 if(wasAtBottom) div.scrollTop = div.scrollHeight;
            }
        }
    } catch(e) { console.error("Log fetch failed", e); }
}

// --- DATA & SYMBOLS ---
async function loadOptions() {
    console.log("Loading data options...");
    try {
        const res = await fetch('/data/map');
        if(!res.ok) throw new Error(res.statusText);
        
        featureMap = await res.json();
        const options = Object.keys(featureMap).sort();
        
        const html = '<option value="">-- Select Configuration --</option>' + options.map(o => {
            let label = o;
            try {
                const j = JSON.parse(o);
                if (j.train_window && j.test_window) label = `Train:${j.train_window} Test:${j.test_window}`; 
            } catch(e) {}
            return `<option value='${o.replace(/'/g, "&#39;")}'>${label}</option>`;
        }).join('');
        
        // Populate all 3 sections
        SECTIONS.forEach(prefix => {
            const el = $(`${prefix}_data_options`);
            if(el) el.innerHTML = html;
        });
        
    } catch(e) {
        console.error("Failed to load map:", e);
        SECTIONS.forEach(prefix => {
            const el = $(`${prefix}_data_options`);
            if(el) el.innerHTML = '<option>Error loading options</option>';
        });
    }
}

function updateSymbols(prefix) {
    const opt = $(`${prefix}_data_options`).value;
    const symSelect = $(`${prefix}_symbol`);
    const ctxSelects = [1,2,3].map(i => $(`${prefix}_ctx${i}`));
    
    if(!opt) {
        symSelect.innerHTML = '<option value="">--</option>';
        ctxSelects.forEach(s => s.innerHTML = '<option value="">(None)</option>');
        return;
    }
    
    const symbols = featureMap[opt] || [];
    const html = symbols.length 
        ? symbols.map(s => `<option value="${s}">${s}</option>`).join('')
        : '<option value="">No symbols</option>';
    
    symSelect.innerHTML = html;
    ctxSelects.forEach(s => s.innerHTML = '<option value="">(None)</option>' + html);
}

// --- PARENT MODELS & FEATURES ---
async function onParentModelChange(prefix) {
    const pid = $(`${prefix}_parent_model`).value;
    const ui = $(`${prefix}_feature_ui`);
    const tbody = $(`${prefix}_feat_body`);
    
    if(!pid || !ui || !tbody) {
        if(ui) ui.style.display = 'none';
        return;
    }
    
    const m = modelsCache[pid];
    if(!m) return;
    
    // Feature Table
    let metrics = {};
    try { metrics = JSON.parse(m.metrics); } catch(e){}
    
    let rows = [];
    if(metrics.feature_details) {
        rows = Object.entries(metrics.feature_details).map(([feat, det]) => ({
            feat, 
            imp: det.coefficient !== undefined ? det.coefficient : (det.tree_importance || 0)
        }));
    } else if (metrics.feature_importance) {
        rows = Object.entries(metrics.feature_importance).map(([feat, score]) => ({feat, imp: score}));
    }
    
    rows.sort((a,b) => Math.abs(b.imp) - Math.abs(a.imp));
    
    if(rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3">No features found in parent</td></tr>';
    } else {
        const fmt = n => Math.abs(n) < 0.001 ? n.toExponential(2) : n.toFixed(4);
        tbody.innerHTML = rows.map(r => `
            <tr>
                <td style="text-align:center"><input type="checkbox" class="${prefix}-feat-check" value="${r.feat}" checked></td>
                <td>${r.feat}</td>
                <td style="text-align:right">${fmt(r.imp)}</td>
            </tr>
        `).join('');
    }
    
    ui.style.display = 'block';
}

// --- TRAIN ---
async function trainModel(prefix) {
    const symbol = $(`${prefix}_symbol`).value;
    if(!symbol) return alert("Please select a symbol");

    // Get specific algorithm from dropdown
    const algo = $(`${prefix}_model_type`).value;
    
    // Config
    const contexts = [1,2,3].map(i => {
        const el = $(`${prefix}_ctx${i}`);
        return el ? el.value : '';
    }).filter(x => x && x !== symbol);
    
    const fullSymbol = [symbol, ...contexts].join(',');
    
    // Params Construction
    const params = {};
    
    // Helper to safely get value (prevent crash if element missing)
    const val = (id, def) => { const el = $(id); return el ? el.value : def; };

    // 1. Gather specific inputs based on prefix (UI Section)
    if(prefix === 'enet') {
        params.alpha = parseFloat(val('enet_alpha', '1.0'));
        params.l1_ratio = parseFloat(val('enet_l1_ratio', '0.5'));
    } else if (prefix === 'xgb') {
        params.n_estimators = parseInt(val('xgb_n_estimators', '100'));
        params.max_depth = parseInt(val('xgb_max_depth', '6'));
        params.learning_rate = parseFloat(val('xgb_learning_rate', '0.1'));
        params.subsample = parseFloat(val('xgb_subsample', '1.0'));
        params.colsample_bytree = parseFloat(val('xgb_colsample_bytree', '1.0'));
        params.min_child_weight = parseInt(val('xgb_min_child_weight', '1'));
        params.reg_alpha = parseFloat(val('xgb_reg_alpha', '0.0'));
        params.reg_lambda = parseFloat(val('xgb_reg_lambda', '1.0'));
        
        const residBase = val('xgb_residual_base_model', '');
        if(residBase) params.residual_base_model_id = residBase;

    } else if (prefix === 'lgbm') {
        params.n_estimators = parseInt(val('lgbm_n_estimators', '100'));
        params.learning_rate = parseFloat(val('lgbm_learning_rate', '0.1'));
        params.num_leaves = parseInt(val('lgbm_num_leaves', '31'));
        params.min_child_samples = parseInt(val('lgbm_min_data_in_leaf', '20')); 
        params.feature_fraction = parseFloat(val('lgbm_feature_fraction', '1.0'));
        params.reg_alpha = parseFloat(val('lgbm_lambda_l1', '0.0'));
        params.reg_lambda = parseFloat(val('lgbm_lambda_l2', '0.0'));
    }
    
    // 2. Merge with JSON overrides
    const jsonStr = val(`${prefix}_hyperparameters`, '');
    if(jsonStr.trim()) {
        try {
            const overrides = JSON.parse(jsonStr);
            Object.assign(params, overrides);
        } catch(e) {
            return alert("Invalid JSON in overrides: " + e);
        }
    }

    // 3. Feature Whitelist
    let featureWhitelist = null;
    if(val(`${prefix}_parent_model`, '')) {
        const checked = document.querySelectorAll(`.${prefix}-feat-check:checked`);
        featureWhitelist = Array.from(checked).map(c => c.value);
    }

    const btn = document.querySelector(`button[onclick^="trainModel('${prefix}'"]`);
    const oldText = btn.innerText;
    btn.disabled = true; btn.innerText = "Starting...";

    try {
        const res = await fetch('/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                symbol: fullSymbol,
                algorithm: algo,
                target_col: $(`${prefix}_target`).value,
                data_options: $(`${prefix}_data_options`).value || null,
                timeframe: $(`${prefix}_timeframe`).value,
                parent_model_id: $(`${prefix}_parent_model`).value || null,
                feature_whitelist: featureWhitelist,
                target_transform: $(`${prefix}_target_transform`).value,
                hyperparameters: params
            })
        });
        
        if(!res.ok) {
            const err = await res.json();
            alert('Error: ' + err.detail);
        } else {
            loadModels();
        }
    } catch(e) { alert("Req failed: " + e); }
    finally {
        btn.disabled = false; btn.innerText = oldText;
    }
}

// --- MODEL REGISTRY & REPORTING ---
async function loadModels() {
    try {
        const res = await fetch('/models');
        const models = await res.json();
        const tbody = $('models-body');
        
        // Update Parent Dropdowns for all sections
        const completed = models.filter(m => m.status === 'completed');
        const options = '<option value="">(None)</option>' + 
            completed.map(m => `<option value="${m.id}">${m.name} (${m.algorithm})</option>`).join('');
            
        SECTIONS.forEach(prefix => {
            const sel = $(`${prefix}_parent_model`);
            if(sel && sel.innerHTML !== options) { // Avoid resetting selection if possible, simplified here
                 const val = sel.value;
                 sel.innerHTML = options;
                 sel.value = val; 
            }
        });

        // Update XGBoost Residual Base Dropdown (Filter for Elastic Net)
        const enetModels = completed.filter(m => m.algorithm === 'elastic_net');
        const enetOptions = '<option value="">(None)</option>' + 
            enetModels.map(m => `<option value="${m.id}">${m.name} (${m.timeframe})</option>`).join('');
        
        const xgbResSel = $('xgb_residual_base_model');
        if(xgbResSel && xgbResSel.innerHTML !== enetOptions) {
            const val = xgbResSel.value;
            xgbResSel.innerHTML = enetOptions;
            xgbResSel.value = val;
        }

        // Render Table (simplified flat list for new Dashboard)
        if(!models.length) {
            tbody.innerHTML = '<tr><td colspan="9">No models</td></tr>';
            return;
        }
        
        // Sort by created desc
        models.sort((a,b) => b.created_at.localeCompare(a.created_at));
        
        models.forEach(m => modelsCache[m.id] = m);

        tbody.innerHTML = models.map(m => {
            const statusClass = m.status === 'failed' ? 'failed' : (m.status==='completed'?'completed':'running');
            const gridInfo = getGridInfo(m, models);
            return `
            <tr>
                <td><span class="badge" title="${m.id}">${m.id.substring(0,8)}</span></td>
                <td>${m.name}</td>
                <td>${m.algorithm}</td>
                <td>${m.symbol}</td>
                <td>${m.timeframe||'-'}</td>
                <td><span class="badge ${statusClass}">${m.status}</span></td>
                <td>${gridInfo}</td>
                <td>${m.metrics && m.metrics.length > 5 ? `<button onclick="showReport('${m.id}')" class="secondary">Report</button>` : '-'}</td>
                <td>${m.created_at.split('T')[1].split('.')[0]}</td>
                <td>
                    <button class="secondary" style="color:#fca5a5" onclick="deleteModel('${m.id}')">Del</button>
                    ${m.status === 'completed' ? `<button class="secondary" style="color:#60a5fa" onclick="retrainModel('${m.id}')">Re</button>` : ''}
                </td>
            </tr>`;
        }).join('');
        
    } catch(e) { console.error(e); }
}

async function deleteModel(id) {
    if(!confirm('Delete?')) return;
    await fetch(`/models/${id}`, {method: 'DELETE'});
    loadModels();
}

async function deleteAllModels() {
    if(!confirm('Delete ALL?')) return;
    if(prompt("Type DELETE") !== "DELETE") return;
    await fetch('/models/all', {method: 'DELETE'});
    loadModels();
}

async function retrainModel(id) {
    if(!confirm('Retrain?')) return;
    await fetch(`/retrain/${id}`, {method: 'POST'});
}

function showReport(id) {
    const m = modelsCache[id];
    if(!m) return;
    $('report-title').innerText = m.name;
    // Primitive dump for now
    let html = '<pre>' + JSON.stringify(JSON.parse(m.metrics || '{}'), null, 2) + '</pre>';
    $('report-content').innerHTML = html;
    $('report-modal').showModal();
}
function closeReport() { $('report-modal').close(); }

window.addEventListener("load", load);

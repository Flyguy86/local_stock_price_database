const $ = id => document.getElementById(id);
const modelsCache = {};
const SECTIONS = ['enet', 'xgb', 'lgbm'];
let featureMap = {};

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
async function trainModel(prefix, algo) {
    const symbol = $(`${prefix}_symbol`).value;
    if(!symbol) return alert("Please select a symbol");
    
    // Config
    const contexts = [1,2,3].map(i => $(`${prefix}_ctx${i}`).value).filter(x => x && x !== symbol);
    const fullSymbol = [symbol, ...contexts].join(',');
    
    // Params Construction
    const params = {};
    
    // 1. Gather specific inputs based on algo
    if(algo === 'elastic_net') {
        params.alpha = parseFloat($('enet_alpha').value);
        params.l1_ratio = parseFloat($('enet_l1_ratio').value);
    } else if (algo === 'xgboost') {
        params.n_estimators = parseInt($('xgb_n_estimators').value);
        params.max_depth = parseInt($('xgb_max_depth').value);
        params.learning_rate = parseFloat($('xgb_learning_rate').value);
        // New params
        params.subsample = parseFloat($('xgb_subsample').value);
        params.colsample_bytree = parseFloat($('xgb_colsample_bytree').value);
        params.min_child_weight = parseInt($('xgb_min_child_weight').value);
        
        // Hybrid / Residual Learning
        const residBase = $('xgb_residual_base_model').value;
        if(residBase) {
            params.residual_base_model_id = residBase;
        }

    } else if (algo === 'lightgbm') {
        params.n_estimators = parseInt($('lgbm_n_estimators').value);
        params.learning_rate = parseFloat($('lgbm_learning_rate').value);
        params.num_leaves = parseInt($('lgbm_num_leaves').value);
        // New params
        params.min_child_samples = parseInt($('lgbm_min_data_in_leaf').value); // Maps to min_data_in_leaf
        params.feature_fraction = parseFloat($('lgbm_feature_fraction').value);
        params.reg_alpha = parseFloat($('lgbm_lambda_l1').value);   // Maps to lambda_l1
        params.reg_lambda = parseFloat($('lgbm_lambda_l2').value); // Maps to lambda_l2
    }
    
    // 2. Merge with JSON overrides
    const jsonStr = $(`${prefix}_hyperparameters`).value.trim();
    if(jsonStr) {
        try {
            const overrides = JSON.parse(jsonStr);
            Object.assign(params, overrides);
        } catch(e) {
            return alert("Invalid JSON in overrides: " + e);
        }
    }

    // 3. Feature Whitelist
    let featureWhitelist = null;
    if($(`${prefix}_parent_model`).value) {
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
            return `
            <tr>
                <td><span class="badge" title="${m.id}">${m.id.substring(0,8)}</span></td>
                <td>${m.name}</td>
                <td>${m.algorithm}</td>
                <td>${m.symbol}</td>
                <td>${m.timeframe||'-'}</td>
                <td><span class="badge ${statusClass}">${m.status}</span></td>
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

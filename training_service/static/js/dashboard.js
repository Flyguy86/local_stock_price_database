const $ = id => document.getElementById(id);
const modelsCache = {};
const SECTIONS = ['enet', 'xgb', 'lgbm'];
let featureMap = {};

/**
 * Get inline metrics display for table.
 * Shows key performance metrics based on model type.
 */
function getMetricsDisplay(model) {
    if (model.status !== 'completed' || !model.metrics) {
        return '-';
    }
    
    try {
        const metrics = typeof model.metrics === 'string' ? JSON.parse(model.metrics) : model.metrics;
        const isRegression = model.algorithm && (model.algorithm.includes('regressor') || model.algorithm.includes('regression'));
        
        if (isRegression) {
            // Show R² and RMSE for regression
            const r2 = metrics.r2 !== undefined ? metrics.r2.toFixed(3) : '-';
            const rmse = metrics.rmse !== undefined ? metrics.rmse.toFixed(4) : '-';
            return `<div style="font-size:0.85rem; line-height:1.3">
                <div><strong>R²:</strong> ${r2}</div>
                <div><strong>RMSE:</strong> ${rmse}</div>
            </div>`;
        } else {
            // Show Accuracy and F1 for classification
            const acc = metrics.accuracy !== undefined ? (metrics.accuracy * 100).toFixed(1) + '%' : '-';
            const f1 = metrics.f1_score !== undefined ? (metrics.f1_score * 100).toFixed(1) + '%' : '-';
            return `<div style="font-size:0.85rem; line-height:1.3">
                <div><strong>Acc:</strong> ${acc}</div>
                <div><strong>F1:</strong> ${f1}</div>
            </div>`;
        }
    } catch(e) {
        console.error('Error parsing metrics for display:', e);
        return '-';
    }
}

/**
 * Get fingerprint display for table.
 * Shows first 8 chars with full fingerprint on hover.
 */
function getFingerprintDisplay(model) {
    if (!model.fingerprint) {
        return '<span style="color:var(--text-muted); font-size:0.8rem">none</span>';
    }
    
    const short = model.fingerprint.substring(0, 8);
    return `<span class="badge" style="font-family:monospace; font-size:0.7rem; cursor:help" title="${model.fingerprint}">${short}...</span>`;
}

/**
 * Get grid search display info for a model.
 * Shows if model is a cohort leader (with cohort size) or cohort member (with alpha/l1_ratio).
 */
/**
 * Format parent model ID for display
 */
function getParentDisplay(model) {
    if (!model.parent_model_id) {
        return '-';
    }
    
    // Show abbreviated parent ID with tooltip
    const shortId = model.parent_model_id.substring(0, 8);
    return `<span class="badge" style="background: rgba(139, 92, 246, 0.2); color: #a78bfa; cursor: pointer;" 
                  title="Parent: ${model.parent_model_id}\nClick to highlight parent" 
                  onclick="highlightModel('${model.parent_model_id}')">
              ⬆️ ${shortId}
            </span>`;
}

/**
 * Format cohort ID for display
 */
function getCohortDisplay(model) {
    if (!model.cohort_id) {
        return '-';
    }
    
    // Show abbreviated cohort ID with tooltip and sibling count
    const shortId = model.cohort_id.substring(0, 8);
    const siblingCount = model.cohort_size || 0;
    
    return `<span class="badge" style="background: rgba(244, 114, 182, 0.2); color: #f472b6; cursor: pointer;" 
                  title="Cohort: ${model.cohort_id}\nSiblings: ${siblingCount}\nClick to see cohort details" 
                  onclick="showGridDetails('${model.cohort_id}')">
              🤝 ${shortId}
            </span>`;
}

/**
 * Highlight a model row temporarily
 */
function highlightModel(modelId) {
    // Find all table rows
    const rows = document.querySelectorAll('#models-body tr');
    
    rows.forEach(row => {
        const idBadge = row.querySelector('.badge');
        if (idBadge && idBadge.title === modelId) {
            // Flash highlight
            row.style.backgroundColor = 'rgba(139, 92, 246, 0.3)';
            row.style.transition = 'background-color 0.3s';
            
            // Scroll into view
            row.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Remove highlight after 2 seconds
            setTimeout(() => {
                row.style.backgroundColor = '';
            }, 2000);
        }
    });
}

function getGridInfo(model, allModels) {
    // NEW COHORT SYSTEM (with cohort_id column)
    const cohortSize = model.cohort_size || 0;
    if (cohortSize > 0 && model.cohort_id) {
        return `<span class="badge grid-parent grid-indicator" style="cursor:pointer" onclick="showGridDetails('${model.cohort_id}')" title="Click to see cohort details">🔍 <span class="count">${cohortSize}</span> siblings ✓</span>`;
    }
    
    if (model.is_grid_member && model.cohort_id) {
        const paramStr = formatModelParams(model);
        return `<span class="badge grid-child" style="cursor:pointer" onclick="showGridDetails('${model.cohort_id}')" title="${paramStr}">🤝 ${paramStr}</span>`;
    }
    
    // LEGACY SYSTEM (with parent_model_id) - backward compatibility
    // Check if this model has grid children using parent_model_id
    const gridChildrenCount = model.grid_children_count || 0;
    if (gridChildrenCount > 0 && model.parent_model_id === null) {
        return `<span class="badge grid-parent grid-indicator" style="cursor:pointer" title="Grid parent with ${gridChildrenCount} children">🔍 <span class="count">${gridChildrenCount}</span> children</span>`;
    }
    
    // Check if this is a grid child (has parent_model_id)
    if (model.parent_model_id) {
        const paramStr = formatModelParams(model);
        return `<span class="badge grid-child" title="${paramStr}">🤝 ${paramStr}</span>`;
    }
    
    // Display params for ALL models that have hyperparameters
    if (model.hyperparameters) {
        const paramStr = formatModelParams(model);
        console.log(`Grid display for ${model.id}: "${paramStr}"`);
        
        // Show any valid parameter string
        if (paramStr && paramStr !== 'no params' && paramStr !== 'parse error' && paramStr !== 'error') {
            return `<span style="font-size:0.75rem; color:var(--text-muted)">${paramStr}</span>`;
        }
    }
    
    console.log(`Grid column for ${model.id}: no params to display`);
    return '-';
}

/**
 * Format model hyperparameters for display based on algorithm.
 */
function formatModelParams(model) {
    try {
        let hp = model.hyperparameters;
        
        // Parse if it's a JSON string
        if (typeof hp === 'string') {
            try {
                hp = JSON.parse(hp);
            } catch(e) {
                console.error('Failed to parse hyperparameters JSON:', hp, e);
                return 'parse error';
            }
        }
        
        if (!hp || typeof hp !== 'object') {
            console.warn('No valid hyperparameters for model:', model.id, 'hp:', hp);
            return 'no params';
        }
        
        const algo = model.algorithm;
        const parts = [];
        
        // ElasticNet / Ridge
        if (algo === 'elastic_net' || algo === 'elasticnet_regression' || algo === 'ridge') {
            if (hp.alpha !== undefined) parts.push(`α=${hp.alpha}`);
            if (hp.l1_ratio !== undefined) parts.push(`L1=${hp.l1_ratio}`);
        }
        // XGBoost
        else if (algo === 'xgboost') {
            if (hp.max_depth !== undefined) parts.push(`depth=${hp.max_depth}`);
            if (hp.learning_rate !== undefined) parts.push(`lr=${hp.learning_rate}`);
            if (hp.n_estimators !== undefined) parts.push(`trees=${hp.n_estimators}`);
            if (hp.subsample !== undefined) parts.push(`sub=${hp.subsample}`);
        }
        // LightGBM
        else if (algo === 'lightgbm') {
            if (hp.max_depth !== undefined) parts.push(`depth=${hp.max_depth}`);
            if (hp.learning_rate !== undefined) parts.push(`lr=${hp.learning_rate}`);
            if (hp.n_estimators !== undefined) parts.push(`trees=${hp.n_estimators}`);
            if (hp.num_leaves !== undefined) parts.push(`leaves=${hp.num_leaves}`);
        }
        // Random Forest
        else if (algo === 'random_forest') {
            if (hp.max_depth !== undefined) parts.push(`depth=${hp.max_depth}`);
            if (hp.n_estimators !== undefined) parts.push(`trees=${hp.n_estimators}`);
            if (hp.min_samples_split !== undefined) parts.push(`split=${hp.min_samples_split}`);
        }
        // Generic fallback - show first 3 params
        else {
            const keys = Object.keys(hp).slice(0, 3);
            keys.forEach(k => {
                let val = hp[k];
                if (typeof val === 'number') val = val.toFixed(4).replace(/\.?0+$/, '');
                parts.push(`${k}=${val}`);
            });
        }
        
        const result = parts.length > 0 ? parts.join(' ') : '';
        console.log(`formatModelParams for ${model.id} (${algo}):`, result, 'from hp:', hp);
        return result || 'no params';
    } catch(e) {
        console.error('Error formatting params for model', model.id, ':', e);
        return 'error';
    }
}

/**
 * Show grid search cohort details in a modal.
 * Displays cohort info and all sibling models in a table.
 */
async function showGridDetails(cohortId) {
    // Find all models in this cohort
    const cohortModels = Object.values(modelsCache).filter(m => m.cohort_id === cohortId);
    
    if (cohortModels.length === 0) {
        alert('No models found in this cohort');
        return;
    }
    
    // Pick first model as representative for cohort metadata
    const representative = cohortModels[0];
    
    // Debug logging
    console.log(`Found ${cohortModels.length} models in cohort ${cohortId}`);
    if (cohortModels.length > 0) {
        console.log('First model sample:', cohortModels[0]);
        console.log('First model hyperparameters:', cohortModels[0].hyperparameters);
        console.log('Hyperparameters type:', typeof cohortModels[0].hyperparameters);
    }
    
    cohortModels.sort((a, b) => {
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
            <h3 style="margin:0 0 0.5rem 0; color:#a78bfa">🤝 Grid Search Cohort</h3>
            <div style="display:grid; grid-template-columns:auto 1fr; gap:0.25rem 1rem; font-size:0.85rem">
                <span style="color:var(--text-muted)">Cohort ID:</span><span style="font-family:monospace">${cohortId.substring(0, 12)}...</span>
                <span style="color:var(--text-muted)">Symbol:</span><span>${representative.symbol}</span>
                <span style="color:var(--text-muted)">Algorithm:</span><span>${representative.algorithm}</span>
                <span style="color:var(--text-muted)">Siblings:</span><span>${cohortModels.length}</span>
            </div>
        </div>
    `;
    
    // Grid search verification status
    const completedModels = cohortModels.filter(c => c.status === 'completed').length;
    const failedModels = cohortModels.filter(c => c.status === 'failed').length;
    const totalModels = cohortModels.length;
    
    const isSuccess = completedModels === totalModels && totalModels > 0;
    const hasIssues = failedModels > 0;
    
    html += `
        <div style="margin-bottom:1rem; padding:0.75rem; background:${isSuccess ? 'rgba(16,185,129,0.1)' : hasIssues ? 'rgba(239,68,68,0.1)' : 'rgba(59,130,246,0.1)'}; border-radius:6px; border:1px solid ${isSuccess ? 'rgba(16,185,129,0.3)' : hasIssues ? 'rgba(239,68,68,0.3)' : 'rgba(59,130,246,0.3)'}">
            <h3 style="margin:0 0 0.5rem 0; color:${isSuccess ? '#34d399' : hasIssues ? '#fca5a5' : '#60a5fa'}">${isSuccess ? '✅ Grid Search Complete' : hasIssues ? '❌ Grid Search Has Issues' : '⏳ Grid Search In Progress'}</h3>
            <div style="font-size:0.85rem">
                <span style="color:var(--text-muted)">Total Models:</span> <strong>${totalModels}</strong><br>
                <span style="color:var(--text-muted)">Completed:</span> <strong style="color:#34d399">${completedModels}</strong><br>
                ${failedModels > 0 ? `<span style="color:var(--text-muted)">Failed:</span> <strong style="color:#fca5a5">${failedModels}</strong><br>` : ''}
            </div>
        </div>
    `;
    
    // Cohort siblings table
    if (cohortModels.length > 0) {
        // Determine which columns to show based on algorithm
        const algo = representative.algorithm;
        let paramColumns = [];
        
        if (algo === 'elastic_net' || algo === 'elasticnet_regression') {
            paramColumns = [{key: 'alpha', label: 'Alpha (α)'}, {key: 'l1_ratio', label: 'L1 Ratio'}];
        } else if (algo === 'ridge') {
            paramColumns = [{key: 'alpha', label: 'Alpha (α)'}];
        } else if (algo === 'xgboost') {
            paramColumns = [
                {key: 'max_depth', label: 'Max Depth'},
                {key: 'learning_rate', label: 'Learn Rate'},
                {key: 'n_estimators', label: 'Trees'},
                {key: 'subsample', label: 'Subsample'}
            ];
        } else if (algo === 'lightgbm') {
            paramColumns = [
                {key: 'max_depth', label: 'Max Depth'},
                {key: 'learning_rate', label: 'Learn Rate'},
                {key: 'n_estimators', label: 'Trees'},
                {key: 'num_leaves', label: 'Leaves'}
            ];
        } else if (algo === 'random_forest') {
            paramColumns = [
                {key: 'max_depth', label: 'Max Depth'},
                {key: 'n_estimators', label: 'Trees'},
                {key: 'min_samples_split', label: 'Min Split'}
            ];
        } else {
            // Generic: show first 3 params from first model
            try {
                const sampleHp = typeof cohortModels[0].hyperparameters === 'string' 
                    ? JSON.parse(cohortModels[0].hyperparameters) 
                    : cohortModels[0].hyperparameters;
                if (sampleHp) {
                    paramColumns = Object.keys(sampleHp).slice(0, 3).map(k => ({key: k, label: k}));
                }
            } catch(e) {}
        }
        
        html += `
            <h3 style="margin:0 0 0.5rem 0">🤝 Cohort Siblings (${cohortModels.length})</h3>
            <div style="max-height:300px; overflow-y:auto">
                <table style="width:100%; font-size:0.8rem">
                    <thead>
                        <tr>
                            ${paramColumns.map(col => `<th>${col.label}</th>`).join('')}
                            <th>Status</th>
                            <th>R² Score</th>
                            <th>ID</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        for (const sibling of cohortModels) {
            let hp = {};
            
            // Try to parse hyperparameters
            try { 
                hp = typeof sibling.hyperparameters === 'string' ? JSON.parse(sibling.hyperparameters) : (sibling.hyperparameters || {}); 
            } catch(e){
                console.error('Failed to parse hyperparameters for sibling', sibling.id, e);
                console.log('Raw hyperparameters:', sibling.hyperparameters);
            }
            
            // Extract parameter values
            const paramValues = paramColumns.map(col => {
                const val = hp[col.key];
                if (val === undefined || val === null) return '-';
                if (typeof val === 'number') return val.toFixed(4).replace(/\.?0+$/, '');
                return val;
            });
            
            let r2 = '-';
            try {
                let metrics = typeof sibling.metrics === 'string' ? JSON.parse(sibling.metrics) : sibling.metrics;
                if (metrics) {
                    r2 = metrics.r2 || metrics.R2 || metrics.test_r2 || metrics.cv_score || metrics.cv_mean_r2 || '-';
                    if (typeof r2 === 'number') r2 = r2.toFixed(4);
                }
            } catch(e){}
            
            const statusClass = sibling.status === 'completed' ? 'completed' : sibling.status === 'failed' ? 'failed' : 'running';
            
            html += `
                <tr>
                    ${paramValues.map(val => `<td>${val}</td>`).join('')}
                    <td><span class="badge ${statusClass}">${sibling.status}</span></td>
                    <td>${r2}</td>
                    <td><span class="badge" title="${sibling.id}">${sibling.id.substring(0,8)}</span></td>
                </tr>
            `;
        }
        
        html += `</tbody></table></div>`;
    } else {
        html += `<p style="color:var(--text-muted)">No cohort models found yet. Grid search may still be running.</p>`;
    }
    
    $('grid-title').innerText = `🤝 Grid Search Cohort: ${representative.symbol} ${representative.algorithm}`;
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
            const result = await res.json();
            
            // Check if duplicate was detected
            if (result.status === 'duplicate') {
                const msg = `⚠️ Duplicate Detected!\n\n${result.message}\n\nExisting Model ID: ${result.existing_model.id}\nStatus: ${result.existing_model.status}\n\nNo new training started.`;
                alert(msg);
                console.log('Duplicate model found:', result.existing_model);
            } else {
                console.log('New training started:', result.id);
            }
            
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
        
        // Debug: Log first 3 models to see structure
        if (models.length > 0) {
            console.log('=== Model Data Debug ===');
            models.slice(0, 3).forEach((m, i) => {
                console.log(`Model ${i+1}:`, {
                    id: m.id,
                    algorithm: m.algorithm,
                    cohort_id: m.cohort_id,
                    is_grid_member: m.is_grid_member,
                    parent_model_id: m.parent_model_id,
                    grid_children_count: m.grid_children_count,
                    cohort_size: m.cohort_size,
                    hyperparameters_type: typeof m.hyperparameters,
                    hyperparameters: m.hyperparameters
                });
            });
        }
        
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
            tbody.innerHTML = '<tr><td colspan="13">No models</td></tr>';
            return;
        }
        
        // Sort by created desc
        models.sort((a,b) => b.created_at.localeCompare(a.created_at));
        
        models.forEach(m => modelsCache[m.id] = m);

        tbody.innerHTML = models.map(m => {
            const statusClass = m.status === 'failed' ? 'failed' : (m.status==='completed'?'completed':'running');
            const gridInfo = getGridInfo(m, models);
            const metricsDisplay = getMetricsDisplay(m);
            const fingerprintDisplay = getFingerprintDisplay(m);
            const parentDisplay = getParentDisplay(m);
            const cohortDisplay = getCohortDisplay(m);
            return `
            <tr>
                <td><span class="badge" title="${m.id}">${m.id.substring(0,8)}</span></td>
                <td>${m.name}</td>
                <td>${m.algorithm}</td>
                <td>${m.symbol}</td>
                <td>${m.timeframe||'-'}</td>
                <td><span class="badge ${statusClass}">${m.status}</span></td>
                <td>${parentDisplay}</td>
                <td>${cohortDisplay}</td>
                <td>${gridInfo}</td>
                <td>${metricsDisplay}</td>
                <td>${fingerprintDisplay}</td>
                <td>${m.created_at.split('T')[1].split('.')[0]}</td>
                <td>
                    <button class="secondary" style="color:#fca5a5" onclick="deleteModel('${m.id}')">Del</button>
                    ${m.status === 'completed' && m.metrics ? `<button class="secondary" style="color:#60a5fa" onclick="showReport('${m.id}')">📊</button>` : ''}
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
    
    // Parse metrics
    let metrics = {};
    try {
        metrics = typeof m.metrics === 'string' ? JSON.parse(m.metrics) : (m.metrics || {});
    } catch(e) {
        console.error('Failed to parse metrics:', e);
    }
    
    const isRegression = m.algorithm && (m.algorithm.includes('regressor') || m.algorithm.includes('regression'));
    const isClassification = !isRegression;
    
    let html = '';
    
    // === REGRESSION METRICS ===
    if (isRegression && (metrics.r2 !== undefined || metrics.mse !== undefined)) {
        html += `
            <div style="background:rgba(16,185,129,0.1); padding:1rem; border-radius:6px; margin-bottom:1rem">
                <h3 style="margin:0 0 0.75rem 0; color:#34d399">📊 Regression Performance</h3>
                <div style="display:grid; grid-template-columns:auto 1fr; gap:0.5rem 1.5rem; font-size:0.9rem">
                    ${metrics.r2 !== undefined ? `<strong>R² Score:</strong><span>${metrics.r2.toFixed(4)}</span>` : ''}
                    ${metrics.mae !== undefined ? `<strong>MAE:</strong><span>${metrics.mae.toFixed(6)}</span>` : ''}
                    ${metrics.mse !== undefined ? `<strong>MSE:</strong><span>${metrics.mse.toFixed(6)}</span>` : ''}
                    ${metrics.rmse !== undefined ? `<strong>RMSE:</strong><span>${metrics.rmse.toFixed(6)}</span>` : ''}
                    ${metrics.rmse_price !== undefined ? `<strong>RMSE (Price):</strong><span style="color:#34d399; font-weight:600">$${metrics.rmse_price.toFixed(2)}</span>` : ''}
                </div>
            </div>
        `;
    }
    
    // === CLASSIFICATION METRICS ===
    if (isClassification && (metrics.accuracy !== undefined)) {
        html += `
            <div style="background:rgba(139,92,246,0.1); padding:1rem; border-radius:6px; margin-bottom:1rem">
                <h3 style="margin:0 0 0.75rem 0; color:#a78bfa">🎯 Classification Performance</h3>
                <div style="display:grid; grid-template-columns:auto 1fr; gap:0.5rem 1.5rem; font-size:0.9rem">
                    ${metrics.accuracy !== undefined ? `<strong>Accuracy:</strong><span>${(metrics.accuracy * 100).toFixed(2)}%</span>` : ''}
                    ${metrics.precision !== undefined ? `<strong>Precision:</strong><span>${(metrics.precision * 100).toFixed(2)}%</span>` : ''}
                    ${metrics.recall !== undefined ? `<strong>Recall:</strong><span>${(metrics.recall * 100).toFixed(2)}%</span>` : ''}
                    ${metrics.f1_score !== undefined ? `<strong>F1-Score:</strong><span>${(metrics.f1_score * 100).toFixed(2)}%</span>` : ''}
                </div>
            </div>
        `;
        
        // Confusion Matrix
        if (metrics.confusion_matrix) {
            html += `
                <div style="background:rgba(244,114,182,0.1); padding:1rem; border-radius:6px; margin-bottom:1rem">
                    <h3 style="margin:0 0 0.75rem 0; color:#f472b6">🔢 Confusion Matrix</h3>
                    <table style="font-size:0.85rem; border-collapse:collapse">
                        <thead>
                            <tr>
                                <th></th>
                                ${metrics.confusion_matrix[0].map((_, i) => `<th>Pred ${i}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
                            ${metrics.confusion_matrix.map((row, i) => `
                                <tr>
                                    <td><strong>True ${i}</strong></td>
                                    ${row.map(val => `<td style="text-align:center; padding:0.25rem 0.5rem">${val}</td>`).join('')}
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }
    }
    
    // === CROSS-VALIDATION DETAILS ===
    if (metrics.cv_detail && metrics.cv_detail.length > 0) {
        html += `
            <div style="background:rgba(59,130,246,0.1); padding:1rem; border-radius:6px; margin-bottom:1rem">
                <h3 style="margin:0 0 0.75rem 0; color:#60a5fa">🔄 Cross-Validation (${metrics.cv_folds} folds)</h3>
                <div style="max-height:150px; overflow-y:auto; font-size:0.85rem">
                    ${metrics.cv_detail.map((fold, i) => `
                        <div style="padding:0.25rem 0">
                            <strong>Fold ${i+1}:</strong> 
                            ${fold.score !== undefined ? `Score: ${fold.score.toFixed(4)}` : ''}
                            ${fold.mse !== undefined ? `MSE: ${fold.mse.toFixed(6)}` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    // === RAW JSON (collapsed) ===
    html += `
        <details style="margin-top:1rem">
            <summary style="cursor:pointer; font-weight:600; padding:0.5rem; background:rgba(0,0,0,0.2); border-radius:4px">
                📄 Raw Metrics JSON
            </summary>
            <pre style="margin-top:0.5rem; font-size:0.75rem">${JSON.stringify(metrics, null, 2)}</pre>
        </details>
    `;
    
    $('report-content').innerHTML = html;
    $('report-modal').showModal();
}
function closeReport() { $('report-modal').close(); }

window.addEventListener("load", load);

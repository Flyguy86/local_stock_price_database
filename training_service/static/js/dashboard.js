const $ = id => document.getElementById(id);
const modelsCache = {};

function load() {
    console.log("Dashbard load() started");
    try {
        // Load components in parallel so one slow service doesn't block the UI
        loadAlgos().catch(e => console.error("loadAlgos error:", e));
        loadOptions().catch(e => console.error("loadOptions error:", e));
        loadModels().catch(e => console.error("loadModels error:", e));
        updateLogs().catch(e => console.error("updateLogs error:", e));
        
        setInterval(() => {
            updateLogs().catch(e => console.error("Polling updateLogs error:", e));
        }, 3000); // Poll logs every 3s
        setInterval(() => {
                loadModels().catch(e => console.error("Polling loadModels error:", e));
        }, 5000);

    } catch(e) {
        console.error("Critical error in load():", e);
        alert("Dashboard failed to initialize: " + e);
    }
}

async function updateLogs() {
    try {
        const res = await fetch('/logs');
        if(res.ok) {
            const logs = await res.json();
            const div = $('log-container');
            const wasAtBottom = div.scrollTop + div.clientHeight >= div.scrollHeight - 20;
            
            if(logs.length === 0) {
                div.innerText = "No logs yet.";
            } else {
                div.innerText = logs.join('\n');
            }
            
            if(wasAtBottom) div.scrollTop = div.scrollHeight;
        }
    } catch(e) { 
        console.error("Log fetch failed", e);
        const div = $('log-container');
        if(div) div.innerText = "Error: " + e;
    }
}

async function loadAlgos() {
    try {
        const res = await fetch('/algorithms');
        if(!res.ok) throw new Error(res.statusText);
        const algos = await res.json();
        $('algo').innerHTML = algos.map(a => `<option value="${a}">${a}</option>`).join('');
    } catch(e) {
        console.error("Algos load failed:", e);
        $('algo').innerHTML = '<option>Error loading algos</option>';
    }
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
            feat, shap: 0, perm: 0, coeff: score
        }));
        // Sort by score
        rows.sort((a,b) => b.coeff - a.coeff);
    } else {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding:1rem">No feature details available in parent</td></tr>';
        ui.style.display = 'block';
        return;
    }
    
    const fmt = n => Math.abs(n) < 0.0001 && n !== 0 ? n.toExponential(2) : n.toFixed(5);
    
    tbody.innerHTML = rows.map(r => `
        <tr style="border-bottom: 1px solid var(--border);" class="feat-row" data-shap="${r.shap}" data-coeff="${r.coeff}" data-perm="${r.perm}">
            <td style="text-align:center;">
                <input type="checkbox" class="feat-check" value="${r.feat}" checked onchange="updateCount()">
            </td>
            <td style="padding:0.4rem; font-family:monospace; font-size:0.8rem">
                ${r.feat}
            </td>
            <td style="padding:0.4rem; text-align:right; font-family:monospace; color:${r.shap > 0 ? '#f472b6' : '#94a3b8'}">${fmt(r.shap)}</td>
            <td style="padding:0.4rem; text-align:right; font-family:monospace; color:${r.perm > 0 ? '#4ade80' : '#94a3b8'}">${fmt(r.perm)}</td>
            <td style="padding:0.4rem; text-align:right; font-family:monospace; color:${Math.abs(r.coeff) > 0 ? '#60a5fa' : '#94a3b8'}">${fmt(r.coeff)}</td>
        </tr>
    `).join('');
    
    $('total-count').innerText = rows.length;
    
    ui.style.display = 'block';
    updateCount();
}

function addByImp() {
    const rows = document.querySelectorAll('.feat-row');
    rows.forEach(r => {
        const coeff = parseFloat(r.dataset.coeff || 0);
        const cb = r.querySelector('.feat-check');
        if (cb && coeff !== 0) {
            cb.checked = true;
        }
    });
    updateCount();
}

function addByPerm() {
    const rows = document.querySelectorAll('.feat-row');
    rows.forEach(r => {
        const perm = parseFloat(r.dataset.perm || 0);
        const cb = r.querySelector('.feat-check');
        if (cb && perm !== 0) {
            cb.checked = true;
        }
    });
    updateCount();
}

function toggleFeatures(state) {
    const rows = document.querySelectorAll('.feat-row');
    rows.forEach(r => {
        const cb = r.querySelector('.feat-check');
        if (cb) cb.checked = state;
    });
    updateCount();
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
        console.log("Loading data options...");
        const startTime = performance.now();
        const select = $('data_options');
        
        try {
        const res = await fetch('/data/map');
        if(!res.ok) throw new Error(res.statusText);
        
        featureMap = await res.json();
        const duration = (performance.now() - startTime).toFixed(0);
        console.log(`Data options loaded in ${duration}ms`, featureMap);
        
        const options = Object.keys(featureMap).sort();
        
        if (options.length === 0) {
                select.innerHTML = '<option value="">(No features found)</option>';
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
            // Escape single quotes for HTML attribute
            const safeValue = o.replace(/'/g, "&#39;");
            return `<option value='${safeValue}'>${label}</option>`;
        }).join('');
        
        } catch(e) {
            console.error("Failed to load data map:", e);
            select.innerHTML = '<option value="">Error Loading Data</option>';
            alert("Error loading data options from Feature Builder: " + e);
        }
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
    
    // Update Title
    const titleEl = $('report-title');
    if(titleEl) titleEl.innerText = `Report: ${m.name} (${m.algorithm})`;
    
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
        'mse': { title: 'Mean Squared Error', desc: 'Average squared difference between predicted and actual values. Lower is better.' },
        'rmse': { title: 'RMSE (Model Unit)', desc: 'Error in model units (e.g. Log Return or Price). If < 0.01 with Log Return, it is normal.' },
        'rmse_price': { title: 'Reconstructed Price RMSE ($)', desc: 'Actual dollar error. Useful for comparing Log Return models against Raw Price models.' },
        'accuracy': { title: 'Accuracy Score', desc: 'Percentage of correct predictions.' },
        'features_count': { title: 'Features Used', desc: 'Number of input variables used by the model.' }
    };

    // Prioritize standard metrics
    const priority = ['mse', 'rmse', 'rmse_price', 'accuracy', 'features_count'];
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

async function deleteAllModels() {
    if(!confirm('⚠️ DANGER: Are you sure you want to DELETE ALL models?\n\nThis action cannot be undone.')) return;
    
    const verification = prompt("Type 'DELETE' to confirm wiping all models:");
    if(verification !== 'DELETE') {
        alert("Deletion cancelled. You must type 'DELETE' exactly.");
        return;
    }
    
    const btn = document.querySelector('button[onclick="deleteAllModels()"]');
    const originalText = btn.innerText;
    btn.innerText = "Deleting...";
    btn.disabled = true;

    try {
        const res = await fetch('/models/all', { method: 'DELETE' });
        if(res.ok) {
            alert("All models deleted successfully.");
            loadModels();
        } else {
            const err = await res.json();
            alert('Failed to delete all: ' + err.detail);
        }
    } catch(e) { console.error(e); alert("Error: " + e); }
    finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
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
    
    // Link Sort
    models.forEach(m => {
        if(m.parent_model_id && map[m.parent_model_id]) {
            map[m.parent_model_id].children.push(m);
        } else {
            roots.push(m);
        }
    });
    
    // Recursive Render
    let html = '';

    function renderList(nodes, depth) {
        // Group by ID
        const groups = {}; 
        const singles = [];
        nodes.forEach(n => {
            if(n.group_id) {
                if(!groups[n.group_id]) groups[n.group_id] = [];
                groups[n.group_id].push(n);
            } else {
                singles.push(n);
            }
        });
        
        // Combine into render units
        const units = [];
        Object.values(groups).forEach(g => units.push({type: 'group', items: g}));
        singles.forEach(s => units.push({type: 'single', item: s}));
        
        // Sort units by max created_at
        units.sort((a,b) => {
            const getT = (u) => u.type === 'single' ? u.item.created_at : u.items.reduce((mx, i) => i.created_at > mx ? i.created_at : mx, '');
            return getT(b).localeCompare(getT(a));
        });
        
        units.forEach(u => {
            if(u.type === 'single') {
                renderNode(u.item, depth, null);
            } else {
                // Sort inside group by target? 
                // open -> close -> high -> low preference? 
                // simple alpha sort is close, high, low, open. 
                // Let's stick with simple sort or preserve create order.
                // u.items.sort((a,b) => a.target_col.localeCompare(b.target_col));
                
                u.items.forEach((item, idx) => {
                    renderNode(item, depth, {
                        isGroup: true,
                        isFirst: idx === 0,
                        isLast: idx === u.items.length - 1
                    });
                });
            }
        });
    }
    
    function renderNode(m, depth, groupInfo) {
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
        
        // Styling
        const indent = depth * 20;
        const treeIcon = depth > 0 ? `<span style="color:var(--text-muted); margin-right:5px;">↳</span>` : '';
        
        let rowStyle = `vertical-align: top; background: rgba(255,255,255, ${depth * 0.02});`;
        let firstTdStyle = `padding-left: ${10 + indent}px;`;
        
        if(groupInfo && groupInfo.isGroup) {
            rowStyle += `background: rgba(94, 234, 212, 0.03);`; // teal tint
            firstTdStyle += `border-left: 3px solid #5eead4; padding-left: ${7 + indent}px;`; // Compensate padding for border
            
            if(groupInfo.isFirst) {
                // maybe add a top margin spacer invisible row? 
                // html += `<tr style="height:5px;"></tr>`; 
            }
        }

        html += `
        <tr style="${rowStyle}">
            <td style="${firstTdStyle}">
                ${treeIcon}
                <span class="badge" title="${m.id}">${m.id.substring(0,8)}</span>
                ${groupInfo && groupInfo.isGroup ? '<span style="color:#5eead4; font-size:0.7em; margin-left:4px;">(Grp)</span>' : ''}
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
        renderList(m.children, depth + 1);
    };
    
    renderList(roots, 0);
    
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

    // Parse Hyperparameters
    let params = null;
    const paramsStr = $('hyperparameters').value.trim();
    if(paramsStr) {
        try {
            params = JSON.parse(paramsStr);
        } catch(e) {
            alert('Invalid JSON in Hyperparameters field: ' + e);
            if(btn) { btn.disabled = false; btn.innerText = 'Start Single Job'; }
            return;
        }
    }

    const btn = document.querySelector('button[onclick="train()"]');
    if(btn) {
        btn.disabled = true;
        btn.innerText = 'Starting...';
    }
    
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
                feature_whitelist: featureWhitelist,
                target_transform: $('target_transform').value,
                hyperparameters: params
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
        if(btn) {
            btn.disabled = false;
            btn.innerText = 'Start Single Job';
        }
    }
}

async function trainBatch() {
    const symbol = $('symbol').value.trim().toUpperCase();
    if(!symbol) return alert('Symbol required');
    
    const tf_oc = $('tf_oc').value;
    const tf_hl = $('tf_hl').value;

    // Collect context (same as train)
    const ctx = [
        $('ctx1').value,
        $('ctx2').value,
        $('ctx3').value
    ].filter(s => s && s !== symbol);
    
    const fullSymbolString = [symbol, ...ctx].join(',');

    let featureWhitelist = null;
    if($('parent_model').value) {
        const checked = Array.from(document.querySelectorAll('.feat-check:checked')).map(c => c.value);
        featureWhitelist = checked;
    }

    if(!confirm(`Start 4 training jobs for ${symbol}?\n- Open/Close @ ${tf_oc}\n- High/Low @ ${tf_hl}`)) return;

    const btn = document.querySelector('button[onclick="trainBatch()"]'); 
    const originalText = btn ? btn.innerText : 'Train High/Low/Open/Close';
    if(btn) {
            btn.disabled = true;
            btn.innerText = 'Starting...';
    }
    
    try {
        const res = await fetch('/train/batch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                symbol: fullSymbolString,
                algorithm: $('algo').value,
                data_options: $('data_options').value || null,
                parent_model_id: $('parent_model').value || null,
                feature_whitelist: featureWhitelist,
                timeframe_oc: tf_oc,
                timeframe_hl: tf_hl,
                target_transform: $('target_transform').value
            })
        });
        
        if(!res.ok) {
            const err = await res.json();
            alert('Error: ' + err.detail);
        } else {
            const data = await res.json();
            console.log('Batch started:', data);
            loadModels();
        }
    } catch(e) {
        alert('Request failed: ' + e);
    } finally {
        if(btn) {
            btn.disabled = false;
            btn.innerText = originalText;
        }
    }
}

window.addEventListener("load", load);

/**
 * Orchestrator Dashboard JavaScript
 * Handles all UI interactions for the Recursive Strategy Factory
 */

const API = '';  // Same origin

// ============================================
// Auto-refresh
// ============================================

setInterval(() => {
  loadStats();
  loadRuns();
  loadPromoted();
  loadLogs();
}, 5000);

// Initial load
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadRuns();
  loadPromoted();
  loadLogs();
  setupFormHandler();
  setupSymbolSelector();
  setupGridCalculator();
  toggleRegularizationGrid();  // Initialize visibility based on algorithm
  syncGridPruningDisplay();  // Sync grid pruning display with training config
});

// ============================================
// Grid Search Pruning Toggle
// ============================================

function toggleGridPruningInfo() {
  const checkbox = document.getElementById('grid-enable-pruning');
  const info = document.getElementById('grid-pruning-info');
  
  if (checkbox && info) {
    info.style.display = checkbox.checked ? 'block' : 'none';
    // Sync display values when toggled on
    if (checkbox.checked) {
      syncGridPruningDisplay();
    }
  }
}

function syncGridPruningDisplay() {
  const genDisplay = document.getElementById('grid-gen-display');
  const pruneDisplay = document.getElementById('grid-prune-display');
  const minFeatDisplay = document.getElementById('grid-min-feat-display');
  
  const maxGen = document.getElementById('max-generations')?.value || '4';
  const pruneFrac = document.getElementById('prune-fraction')?.value || '25';
  const minFeat = document.getElementById('min-features')?.value || '5';
  
  if (genDisplay) genDisplay.textContent = maxGen;
  if (pruneDisplay) pruneDisplay.textContent = pruneFrac;
  if (minFeatDisplay) minFeatDisplay.textContent = minFeat;
}

// Sync display when training config values change
document.addEventListener('DOMContentLoaded', () => {
  const maxGenInput = document.getElementById('max-generations');
  const pruneFracInput = document.getElementById('prune-fraction');
  const minFeatInput = document.getElementById('min-features');
  
  if (maxGenInput) maxGenInput.addEventListener('input', syncGridPruningDisplay);
  if (pruneFracInput) pruneFracInput.addEventListener('input', syncGridPruningDisplay);
  if (minFeatInput) minFeatInput.addEventListener('input', syncGridPruningDisplay);
});

// ============================================
// Hyperparameter Grid Toggle (Algorithm-Specific)
// ============================================

function toggleRegularizationGrid() {
  const algorithm = document.getElementById('algorithm')?.value || '';
  
  // Get all grid sections
  const regGrid = document.getElementById('regularization-grid');
  const xgboostGrid = document.getElementById('xgboost-grid');
  const lightgbmGrid = document.getElementById('lightgbm-grid');
  const randomforestGrid = document.getElementById('randomforest-grid');
  
  // Hide all grids first
  if (regGrid) regGrid.style.display = 'none';
  if (xgboostGrid) xgboostGrid.style.display = 'none';
  if (lightgbmGrid) lightgbmGrid.style.display = 'none';
  if (randomforestGrid) randomforestGrid.style.display = 'none';
  
  // Show appropriate grid based on algorithm
  const algoLower = algorithm.toLowerCase();
  
  if (algoLower.includes('elasticnet') || algoLower.includes('ridge') || algoLower.includes('lasso')) {
    // ElasticNet/Ridge/Lasso: Alpha and L1 Ratio
    if (regGrid) regGrid.style.display = 'flex';
  } else if (algoLower.includes('xgboost')) {
    // XGBoost: max_depth, min_child_weight, reg_lambda, learning_rate
    if (xgboostGrid) xgboostGrid.style.display = 'flex';
  } else if (algoLower.includes('lightgbm')) {
    // LightGBM: num_leaves, min_data_in_leaf, lambda_l2, learning_rate
    if (lightgbmGrid) lightgbmGrid.style.display = 'flex';
  } else if (algoLower.includes('random_forest') || algoLower.includes('randomforest')) {
    // Random Forest: max_depth, min_samples_split, min_samples_leaf, n_estimators
    if (randomforestGrid) randomforestGrid.style.display = 'flex';
  }
  // Linear Regression has no hyperparameter grids - all grids stay hidden
}

// ============================================
// Grid Size Calculator
// ============================================

function setupGridCalculator() {
  const thresholdsInput = document.getElementById('thresholds');
  const zScoresInput = document.getElementById('z-scores');
  
  thresholdsInput?.addEventListener('input', updateGridSize);
  zScoresInput?.addEventListener('input', updateGridSize);
  updateGridSize();  // Initial calculation
}

function getSelectedRegimeConfigs() {
  // Build regime configs from checkboxes
  const configs = [];
  
  // VIX regimes (each individually)
  for (let i = 0; i <= 3; i++) {
    const cb = document.getElementById(`regime-vix-${i}`);
    if (cb && cb.checked) {
      configs.push({ "regime_vix": [i] });
    }
  }
  
  // GMM regimes (each individually)
  for (let i = 0; i <= 1; i++) {
    const cb = document.getElementById(`regime-gmm-${i}`);
    if (cb && cb.checked) {
      configs.push({ "regime_gmm": [i] });
    }
  }
  
  // No filter option
  const noneBox = document.getElementById('regime-none');
  if (noneBox && noneBox.checked) {
    configs.push({});
  }
  
  // Fallback: if nothing selected, use "no filter"
  if (configs.length === 0) {
    configs.push({});
  }
  
  return configs;
}

function updateRegimeConfigs() {
  // Called when regime checkboxes change - update grid size
  updateGridSize();
}

function updateGridSize() {
  const thresholdsInput = document.getElementById('thresholds');
  const zScoresInput = document.getElementById('z-scores');
  
  const thresholds = (thresholdsInput?.value || '').split(',').filter(t => t.trim()).length;
  const zScores = (zScoresInput?.value || '').split(',').filter(z => z.trim()).length;
  const regimes = getSelectedRegimeConfigs().length;
  
  // Simulation tickers: use selected count, or 1 if none selected (uses training symbol)
  const tickers = selectedSimTickers?.size > 0 ? selectedSimTickers.size : 1;
  
  const total = tickers * thresholds * zScores * regimes;
  const gridSizeEl = document.getElementById('grid-size');
  const gridDimsEl = document.getElementById('grid-dims');
  
  if (gridSizeEl) gridSizeEl.textContent = total;
  if (gridDimsEl) gridDimsEl.textContent = `${tickers} ticker${tickers > 1 ? 's' : ''} × ${thresholds} thresholds × ${zScores} z-scores × ${regimes} regimes`;
}

// ============================================
// Symbol Selection from Feature Service
// ============================================

let availableOptions = [];
let availableSymbols = [];
let selectedReferences = new Set();
let selectedSimTickers = new Set();  // Simulation tickers (separate from training)
let selectedDataOptions = null;

async function loadAvailableOptions() {
  try {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = '⏳ Loading...';
    
    const res = await fetch(`${API}/api/features/options`);
    const options = await res.json();
    
    if (Array.isArray(options) && options.length > 0) {
      availableOptions = options;
      
      // Display options as radio buttons
      document.getElementById('options-selection').innerHTML = `
        <div style="background: rgba(16, 185, 129, 0.2); padding: 0.75rem; border-radius: 4px;">
          <strong>📁 Select Data Fold:</strong>
          <div style="display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.5rem;">
            ${options.map((opt, idx) => `
              <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; padding: 0.4rem; border-radius: 4px; background: rgba(255,255,255,0.08); transition: background 0.2s;">
                <input type="radio" name="data-options" data-option-idx="${idx}" onchange="onDataOptionsChangeByIndex(${idx})">
                <span style="font-size: 0.9rem; font-family: 'Courier New', monospace;">${opt.length > 60 ? opt.substring(0, 60) + '...' : opt}</span>
              </label>
            `).join('')}
          </div>
        </div>
      `;
    } else {
      document.getElementById('options-selection').innerHTML = `
        <div style="background: rgba(239, 68, 68, 0.2); padding: 0.5rem; border-radius: 4px;">
          ❌ No feature data folds found. Run feature generation first.
        </div>
      `;
    }
    
    btn.disabled = false;
    btn.textContent = '🔄 Load Data Folds';
  } catch (e) {
    console.error('Failed to load options:', e);
    document.getElementById('options-selection').innerHTML = `
      <div style="background: rgba(239, 68, 68, 0.2); padding: 0.5rem; border-radius: 4px;">
        ❌ Error: ${e.message}
      </div>
    `;
    event.target.disabled = false;
    event.target.textContent = '🔄 Load Data Folds';
  }
}

function onDataOptionsChangeByIndex(idx) {
  const selectedOption = availableOptions[idx];
  onDataOptionsChange(selectedOption);
}

async function onDataOptionsChange(selectedOption) {
  selectedDataOptions = selectedOption;
  
  // Show loading state
  document.getElementById('symbol-selection').innerHTML = `
    <div style="color: var(--text-muted); padding: 0.5rem;">
      ⏳ Loading symbols...
    </div>
  `;
  
  try {
    const res = await fetch(`${API}/api/features/symbols?options=${encodeURIComponent(selectedOption)}`);
    console.log('Symbols response:', res.status);
    const data = await res.json();
    
    // Handle both array and {symbols: [...]} response formats
    const symbols = Array.isArray(data) ? data : (data.symbols || []);
    
    if (symbols.length > 0) {
      availableSymbols = symbols;
      
      // Populate target symbol dropdown
      const targetSelect = document.getElementById('symbol');
      targetSelect.innerHTML = '<option value="">Select target symbol...</option>' +
        symbols.map(s => `<option value="${s}">${s}</option>`).join('');
      
      document.getElementById('symbol-selection').innerHTML = `
        <div style="background: rgba(16, 185, 129, 0.2); padding: 0.5rem; border-radius: 4px;">
          ✅ ${symbols.length} symbols available: ${symbols.slice(0, 10).join(', ')}${symbols.length > 10 ? '...' : ''}
        </div>
      `;
      
      // Update reference checkboxes and simulation ticker checkboxes
      updateReferenceCheckboxes();
      updateSimTickerCheckboxes();
    } else {
      document.getElementById('symbol-selection').innerHTML = `
        <div style="background: rgba(239, 68, 68, 0.2); padding: 0.5rem; border-radius: 4px;">
          ❌ No symbols found for this fold
        </div>
      `;
    }
  } catch (e) {
    console.error('Failed to load symbols:', e);
    document.getElementById('symbol-selection').innerHTML = `
      <div style="background: rgba(239, 68, 68, 0.2); padding: 0.5rem; border-radius: 4px;">
        ❌ Error: ${e.message}
      </div>
    `;
  }
}

async function loadAvailableSymbols() {
  // Legacy function - redirect to options workflow
  alert('⚠️ Workflow Changed!\n\nPlease:\n1. Load Data Folds first\n2. Select a fold\n3. Symbols will auto-load for that fold');
  document.getElementById('load-options-btn')?.click();
}

function setupSymbolSelector() {
  // Watch target symbol changes to update reference checkboxes
  const targetSelect = document.getElementById('symbol');
  if (targetSelect) {
    targetSelect.addEventListener('change', () => {
      updateReferenceCheckboxes();
      updateSimTickerCheckboxes();
    });
  }
}

function updateReferenceCheckboxes() {
  const targetSymbol = document.getElementById('symbol').value;
  const container = document.getElementById('reference-checkboxes');
  
  if (!targetSymbol || availableSymbols.length === 0) {
    container.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">Load symbols first</span>';
    return;
  }
  
  // Show all symbols except the target
  const otherSymbols = availableSymbols.filter(s => s !== targetSymbol);
  
  container.innerHTML = otherSymbols.map(sym => `
    <label style="display: flex; align-items: center; gap: 0.25rem; cursor: pointer;">
      <input type="checkbox" value="${sym}" onchange="toggleReferenceSymbol('${sym}')">
      <span style="font-size: 0.85rem;">${sym}</span>
    </label>
  `).join('');
  
  // Update container display
  updateReferenceDisplay();
}

function toggleReferenceSymbol(symbol) {
  if (selectedReferences.has(symbol)) {
    selectedReferences.delete(symbol);
  } else {
    selectedReferences.add(symbol);
  }
  updateReferenceDisplay();
}

function updateReferenceDisplay() {
  const container = document.getElementById('reference-symbols-container');
  
  if (selectedReferences.size === 0) {
    container.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No reference symbols selected (optional)</span>';
  } else {
    container.innerHTML = Array.from(selectedReferences).map(sym => `
      <span class="badge" style="cursor: pointer;" onclick="toggleReferenceSymbol('${sym}'); updateReferenceCheckboxes();">
        ${sym} ×
      </span>
    `).join('');
  }
}

// ============================================
// Simulation Tickers (separate from training)
// ============================================

function updateSimTickerCheckboxes() {
  const targetSymbol = document.getElementById('symbol').value;
  const container = document.getElementById('sim-tickers-checkboxes');
  
  if (!targetSymbol || availableSymbols.length === 0) {
    container.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">Load symbols first</span>';
    return;
  }
  
  // Show all symbols (including target)
  container.innerHTML = availableSymbols.map(sym => `
    <label style="display: flex; align-items: center; gap: 0.25rem; cursor: pointer;">
      <input type="checkbox" value="${sym}" ${selectedSimTickers.has(sym) ? 'checked' : ''} onchange="toggleSimTicker('${sym}')">
      <span style="font-size: 0.85rem;">${sym}</span>
    </label>
  `).join('');
  
  updateSimTickerDisplay();
  updateGridSize();
}

function toggleSimTicker(symbol) {
  if (selectedSimTickers.has(symbol)) {
    selectedSimTickers.delete(symbol);
  } else {
    selectedSimTickers.add(symbol);
  }
  updateSimTickerDisplay();
  updateGridSize();
}

function updateSimTickerDisplay() {
  const container = document.getElementById('sim-tickers-container');
  const placeholder = document.getElementById('sim-tickers-placeholder');
  
  if (selectedSimTickers.size === 0) {
    container.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;" id="sim-tickers-placeholder">Using training symbol (select additional tickers below)</span>';
  } else {
    container.innerHTML = Array.from(selectedSimTickers).map(sym => `
      <span class="badge" style="cursor: pointer; background: rgba(59, 130, 246, 0.3);" onclick="toggleSimTicker('${sym}'); updateSimTickerCheckboxes();">
        ${sym} ×
      </span>
    `).join('');
  }
}

// ============================================
// Stats Loading
// ============================================

async function loadStats() {
  try {
    const [runsRes, pendingRes, promotedRes] = await Promise.all([
      fetch(`${API}/runs`),
      fetch(`${API}/jobs/pending`),
      fetch(`${API}/promoted`)
    ]);
    
    const runs = await runsRes.json();
    const pending = await pendingRes.json();
    const promoted = await promotedRes.json();
    
    document.getElementById('stat-runs').textContent = runs.count || 0;
    document.getElementById('stat-pending').textContent = pending.pending || 0;
    document.getElementById('stat-promoted').textContent = promoted.count || 0;
    
    // Count running runs as "workers" for now
    const runningCount = (runs.runs || []).filter(r => r.status === 'RUNNING').length;
    document.getElementById('stat-workers').textContent = runningCount;
  } catch (e) {
    console.error('Failed to load stats:', e);
  }
}

// ============================================
// Runs Loading & Display
// ============================================

async function loadRuns() {
  try {
    const status = document.getElementById('filter-status').value;
    const url = status ? `${API}/runs?status=${status}` : `${API}/runs`;
    const res = await fetch(url);
    const data = await res.json();
    
    const tbody = document.getElementById('runs-table');
    
    if (!data.runs || data.runs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" style="color: var(--text-muted);">No evolution runs yet. Start one above!</td></tr>';
      return;
    }
    
    tbody.innerHTML = data.runs.map(run => {
      const stepStatus = run.step_status || '-';
      
      // Format progress indicators
      const modelProgress = run.models_total > 0 
        ? `${run.models_trained || 0}/${run.models_total}`
        : '-';
      const simProgress = run.simulations_total > 0
        ? `${run.simulations_completed || 0}/${run.simulations_total}`
        : '-';
      
      // Add action buttons based on status
      let actionButton = `<button class="secondary" onclick="event.stopPropagation(); showDetails('${run.id}')">View</button>`;
      
      if (run.status === 'RUNNING' || run.status === 'PENDING') {
        actionButton = `
          <button class="danger" onclick="event.stopPropagation(); stopRun('${run.id}')" title="Stop this evolution run">⏹️ Stop</button>
          <button class="secondary" onclick="event.stopPropagation(); showDetails('${run.id}')">View</button>
        `;
      } else if (run.status === 'STOPPED' || run.status === 'FAILED') {
        actionButton = `
          <button class="primary" onclick="event.stopPropagation(); resumeRun('${run.id}')" title="Resume from generation ${run.current_generation}">▶️ Resume</button>
          <button class="secondary" onclick="event.stopPropagation(); showDetails('${run.id}')">View</button>
        `;
      }
      
      return `
        <tr class="clickable" onclick="showDetails('${run.id}')">
          <td><code>${run.id.substring(0, 8)}...</code></td>
          <td>${run.symbol}</td>
          <td><span class="badge ${run.status.toLowerCase()}">${run.status}</span></td>
          <td style="font-size: 0.85rem; color: var(--text-muted);">${stepStatus}</td>
          <td>${run.current_generation} / ${run.max_generations}</td>
          <td><span style="font-size: 0.9rem;">${modelProgress}</span></td>
          <td><span style="font-size: 0.9rem;">${simProgress}</span></td>
          <td>${run.best_sqn ? run.best_sqn.toFixed(2) : '-'}</td>
          <td>${new Date(run.created_at).toLocaleString()}</td>
          <td>${actionButton}</td>
        </tr>
      `;
    }).join('');
    
    // Update active runs section
    const activeRuns = data.runs.filter(r => r.status === 'RUNNING' || r.status === 'PENDING');
    const activeDiv = document.getElementById('active-runs');
    
    if (activeRuns.length === 0) {
      activeDiv.innerHTML = '<p style="color: var(--text-muted);">No active runs</p>';
    } else {
      activeDiv.innerHTML = activeRuns.map(run => renderActiveRunCard(run)).join('');
    }
  } catch (e) {
    console.error('Failed to load runs:', e);
  }
}

function renderActiveRunCard(run) {
  const stepStatus = run.step_status || '';
  const modelProgress = run.models_total > 0 
    ? `${run.models_trained || 0}/${run.models_total}`
    : '0/0';
  const simProgress = run.simulations_total > 0
    ? `${run.simulations_completed || 0}/${run.simulations_total}`
    : '0/0';
  
  return `
    <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 6px; margin-bottom: 0.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <strong>${run.symbol}</strong>
          <span class="badge ${run.status.toLowerCase()}" style="margin-left: 0.5rem;">${run.status}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
          <code style="font-size: 0.8rem;">${run.id.substring(0, 8)}</code>
          <button class="danger" style="padding: 0.2rem 0.5rem; font-size: 0.75rem;" onclick="event.stopPropagation(); stopRun('${run.id}')">⏹️ Stop</button>
        </div>
      </div>
      <div style="margin-top: 0.5rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted);">
          <span>Generation ${run.current_generation} of ${run.max_generations}</span>
          <span>Best SQN: ${run.best_sqn ? run.best_sqn.toFixed(2) : '-'}</span>
        </div>
        <div class="progress-bar" style="margin-top: 0.25rem;">
          <div class="progress-fill" style="width: ${(run.current_generation / run.max_generations) * 100}%"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem;">
          <span>🧠 Models: ${modelProgress}</span>
          <span>📊 Sims: ${simProgress}</span>
        </div>
        ${stepStatus ? `<div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--primary); font-style: italic;">📍 ${stepStatus}</div>` : ''}
      </div>
    </div>
  `;
}

async function cancelRun(runId) {
  if (!confirm(`Cancel run ${runId.substring(0, 8)}...?`)) return;
  try {
    const res = await fetch(`${API}/runs/${runId}/cancel`, { method: 'POST' });
    if (res.ok) {
      showStatus('Run cancelled', 'success');
      loadRuns();
    } else {
      const data = await res.json();
      showStatus(data.detail || 'Failed to cancel', 'error');
    }
  } catch (e) {
    showStatus('Error cancelling run', 'error');
  }
}

async function resumeRun(runId) {
  if (!confirm(`Resume interrupted run ${runId.substring(0, 8)}...?\n\nThis will continue from the last completed generation.`)) return;
  try {
    const res = await fetch(`${API}/runs/${runId}/resume`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      showStatus(data.message || 'Run resumed', 'success');
      loadRuns();
      // Auto-refresh to show progress
      setTimeout(loadRuns, 3000);
    } else {
      const data = await res.json();
      showStatus(data.detail || 'Failed to resume', 'error');
    }
  } catch (e) {
    showStatus('Error resuming run: ' + e.message, 'error');
  }
}

async function stopRun(runId) {
  if (!confirm(`Stop run ${runId.substring(0, 8)}...?\n\nThis will mark the run as STOPPED and halt further processing.`)) return;
  try {
    const res = await fetch(`${API}/runs/${runId}/stop`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      showStatus('Run stopped', 'success');
      loadRuns();
    } else {
      const data = await res.json();
      showStatus(data.detail || 'Failed to stop', 'error');
    }
  } catch (e) {
    showStatus('Error stopping run: ' + e.message, 'error');
  }
}

async function cleanupStaleRuns() {
  try {
    const res = await fetch(`${API}/runs/cleanup-stale`, { method: 'POST' });
    const data = await res.json();
    if (data.cleaned > 0) {
      showStatus(`Cleaned up ${data.cleaned} stale run(s)`, 'success');
    } else {
      showStatus('No stale runs found', 'info');
    }
    loadRuns();
  } catch (e) {
    showStatus('Error cleaning up stale runs', 'error');
  }
}

// ============================================
// Promoted Models Loading & Display
// ============================================

async function loadPromoted() {
  try {
    const res = await fetch(`${API}/promoted`);
    const data = await res.json();
    
    const container = document.getElementById('promoted-models');
    
    if (!data.promoted || data.promoted.length === 0) {
      container.innerHTML = '<p style="color: var(--text-muted);">No promoted models yet. Keep evolving!</p>';
      return;
    }
    
    container.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Model ID</th>
            <th>Symbol</th>
            <th>SQN</th>
            <th>Profit Factor</th>
            <th>Trades</th>
            <th>Generation</th>
            <th>Promoted At</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${data.promoted.map(m => renderPromotedRow(m)).join('')}
        </tbody>
      </table>
    `;
  } catch (e) {
    console.error('Failed to load promoted:', e);
  }
}

function renderPromotedRow(m) {
  return `
    <tr style="cursor: pointer;" onclick="showPromotedDetail('${m.id}')">
      <td><code>${m.model_id.substring(0, 8)}...</code></td>
      <td>${m.ticker}</td>
      <td style="color: var(--success); font-weight: bold;">${m.sqn.toFixed(2)}</td>
      <td>${m.profit_factor.toFixed(2)}</td>
      <td>${m.trade_count}</td>
      <td>${m.generation}</td>
      <td>${new Date(m.promoted_at).toLocaleString()}</td>
      <td><button class="secondary" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;">View Details</button></td>
    </tr>
  `;
}

// ============================================
// Promoted Model Detail View
// ============================================

async function showPromotedDetail(promotedId) {
  try {
    const res = await fetch(`${API}/promoted/${promotedId}`);
    const data = await res.json();
    
    document.getElementById('promoted-detail').style.display = 'block';
    document.getElementById('promoted-model-id').textContent = (data.model_id?.substring(0, 12) || '') + '...';
    document.getElementById('promoted-sqn').textContent = data.sqn?.toFixed(2) || '-';
    document.getElementById('promoted-pf').textContent = data.profit_factor?.toFixed(2) || '-';
    document.getElementById('promoted-trades').textContent = data.trade_count || '-';
    document.getElementById('promoted-gen').textContent = data.generation || '-';
    document.getElementById('promoted-symbol').textContent = data.ticker || '-';
    document.getElementById('promoted-threshold').textContent = data.threshold || '-';
    
    // Model config (features, hyperparams)
    const config = data.model_config || {};
    const features = config.features || [];
    document.getElementById('promoted-feature-count').textContent = features.length;
    document.getElementById('promoted-features').textContent = features.length > 0 
      ? features.join('\n') 
      : 'No feature data available';
    document.getElementById('promoted-hyperparams').textContent = config.hyperparams 
      ? JSON.stringify(config.hyperparams, null, 2) 
      : 'No hyperparameter data available';
    
    // Regime config
    document.getElementById('promoted-regime').textContent = data.regime_config 
      ? JSON.stringify(data.regime_config, null, 2) 
      : '-';
    
    // Full result
    document.getElementById('promoted-full-result').textContent = data.full_result 
      ? JSON.stringify(data.full_result, null, 2) 
      : '-';
    
    // Lineage
    renderPromotedLineage(data.lineage || []);
    
    // Scroll to details
    document.getElementById('promoted-detail').scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    console.error('Failed to load promoted detail:', e);
    alert('Failed to load model details');
  }
}

function renderPromotedLineage(lineage) {
  const lineageDiv = document.getElementById('promoted-lineage');
  
  if (lineage.length === 0) {
    lineageDiv.innerHTML = '<p style="color: var(--text-muted);">This is a first-generation model (no ancestry)</p>';
    return;
  }
  
  lineageDiv.innerHTML = lineage.map((l, i) => `
    <div class="lineage-card">
      <div class="lineage-header">
        <span class="badge">Gen ${l.generation}</span>
        <code style="font-size: 0.8rem;">${l.parent_model_id?.substring(0, 8) || 'seed'} → ${l.child_model_id?.substring(0, 8)}</code>
      </div>
      <div style="font-size: 0.85rem; color: var(--text-muted);">
        <strong>Reason:</strong> ${l.pruning_reason || 'Initial'}
      </div>
      <div style="font-size: 0.85rem;">
        <span style="color: var(--danger);">Pruned:</span> 
        ${(l.pruned_features || []).slice(0, 5).join(', ')}${(l.pruned_features || []).length > 5 ? ` (+${l.pruned_features.length - 5} more)` : ''}
      </div>
      <div style="font-size: 0.85rem;">
        <span style="color: var(--success);">Remaining:</span> 
        ${(l.remaining_features || []).length} features
      </div>
    </div>
  `).join('');
}

function hidePromotedDetail() {
  document.getElementById('promoted-detail').style.display = 'none';
}

// ============================================
// Run Details View
// ============================================

async function showDetails(runId) {
  try {
    // Fetch both endpoints in parallel
    const [runRes, genRes] = await Promise.all([
      fetch(`${API}/runs/${runId}`),
      fetch(`${API}/runs/${runId}/generations`)
    ]);
    const data = await runRes.json();
    const genData = await genRes.json();
    
    document.getElementById('run-details').style.display = 'block';
    document.getElementById('detail-run-id').textContent = runId.substring(0, 12) + '...';
    document.getElementById('detail-generation').textContent = data.run.current_generation;
    document.getElementById('detail-sqn').textContent = data.run.best_sqn ? data.run.best_sqn.toFixed(2) : '-';
    document.getElementById('detail-jobs').textContent = data.completed_jobs;
    
    // Per-generation breakdown
    renderGenerationBreakdown(genData.generations || []);
    
    // Lineage tree
    renderRunLineage(data.lineage || []);
    
    // Results table
    renderResultsTable(data.results_sample || []);
    
    // Scroll to details
    document.getElementById('run-details').scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    console.error('Failed to load run details:', e);
    alert('Failed to load run details');
  }
}

function renderGenerationBreakdown(generations) {
  const genDiv = document.getElementById('generation-breakdown');
  
  if (generations.length === 0) {
    genDiv.innerHTML = '<p style="color: var(--text-muted);">No generation data yet</p>';
    return;
  }
  
  genDiv.innerHTML = generations.map(g => {
    const isSuccess = g.best_sqn >= 3;
    return `
      <div class="generation-card ${isSuccess ? 'success' : 'pending'}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
          <span class="badge ${g.completed === g.total_jobs ? 'completed' : 'running'}">Generation ${g.generation}</span>
          <span style="font-size: 0.85rem; color: var(--text-muted);">
            ${g.completed}/${g.total_jobs} jobs (${g.pending} pending, ${g.running} running)
          </span>
        </div>
        <div class="generation-stats">
          <div>
            <div class="generation-stat-label">Best SQN</div>
            <div class="generation-stat-value ${isSuccess ? 'success' : ''}">${g.best_sqn?.toFixed(2) || '-'}</div>
          </div>
          <div>
            <div class="generation-stat-label">Best PF</div>
            <div class="generation-stat-value">${g.best_pf?.toFixed(2) || '-'}</div>
          </div>
          <div>
            <div class="generation-stat-label">Features</div>
            <div class="generation-stat-value">${(g.remaining_features || []).length}</div>
          </div>
        </div>
        ${renderPrunedFeatures(g.pruned_features)}
        ${renderTopResults(g.top_results)}
      </div>
    `;
  }).join('');
}

function renderPrunedFeatures(prunedFeatures) {
  if (!prunedFeatures || prunedFeatures.length === 0) return '';
  
  return `
    <div style="font-size: 0.85rem; margin-bottom: 0.5rem;">
      <span style="color: var(--danger);">Pruned:</span> 
      ${prunedFeatures.slice(0, 3).join(', ')}${prunedFeatures.length > 3 ? ` (+${prunedFeatures.length - 3} more)` : ''}
    </div>
  `;
}

function renderTopResults(topResults) {
  if (!topResults || topResults.length === 0) return '';
  
  return `
    <div style="font-size: 0.85rem; color: var(--text-muted);">
      <strong>Top Results:</strong>
      ${topResults.map(r => `
        <div class="result-chip">
          SQN ${r.sqn?.toFixed(2) || '?'} | PF ${r.profit_factor?.toFixed(2) || '?'} | ${r.trade_count || '?'} trades
        </div>
      `).join('')}
    </div>
  `;
}

function renderRunLineage(lineage) {
  const lineageDiv = document.getElementById('lineage-tree');
  
  if (lineage.length === 0) {
    lineageDiv.innerHTML = '<p style="color: var(--text-muted);">No lineage data yet</p>';
    return;
  }
  
  lineageDiv.innerHTML = lineage.map((l, i) => {
    const pruned = typeof l.pruned_features === 'string' ? JSON.parse(l.pruned_features) : (l.pruned_features || []);
    const remaining = typeof l.remaining_features === 'string' ? JSON.parse(l.remaining_features) : (l.remaining_features || []);
    return `
      <div style="display: flex; flex-direction: column; gap: 0.25rem; padding: 0.5rem; background: rgba(0,0,0,0.2); border-radius: 4px; margin-bottom: 0.25rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
          <span class="badge">Gen ${l.generation}</span>
          <span style="color: var(--text-muted);">→</span>
          <code style="font-size: 0.8rem;">${l.child_model_id.substring(0, 8)}</code>
          <span style="color: var(--text-muted); font-size: 0.85rem;">
            (${remaining.length} features, pruned ${pruned.length})
          </span>
        </div>
        ${l.pruning_reason ? `<div style="font-size: 0.8rem; color: var(--text-muted);">${l.pruning_reason}</div>` : ''}
      </div>
    `;
  }).join('');
}

function renderResultsTable(resultsSample) {
  const resultsTable = document.getElementById('results-table');
  
  if (resultsSample.length === 0) {
    resultsTable.innerHTML = '<tr><td colspan="6" style="color: var(--text-muted);">No results yet</td></tr>';
    return;
  }
  
  resultsTable.innerHTML = resultsSample.map(r => {
    const result = typeof r.result === 'string' ? JSON.parse(r.result) : r.result;
    return `
      <tr>
        <td><code>${r.model_id.substring(0, 8)}</code></td>
        <td>${result?.sqn?.toFixed(2) || '-'}</td>
        <td>${result?.profit_factor?.toFixed(2) || '-'}</td>
        <td>${result?.trade_count || result?.trades_count || '-'}</td>
        <td>${result?.params?.threshold || '-'}</td>
        <td>${JSON.stringify(result?.params?.regime_config || '-')}</td>
      </tr>
    `;
  }).join('');
}

function hideDetails() {
  document.getElementById('run-details').style.display = 'none';
}

// ============================================
// Form Handling
// ============================================

// ============================================
// Model Browser & Selection for Simulations
// ============================================

let availableModels = [];
let selectedModelIds = new Set();

async function toggleModelBrowser() {
  const panel = document.getElementById('model-browser-panel');
  const isVisible = panel.style.display !== 'none';
  
  if (isVisible) {
    panel.style.display = 'none';
  } else {
    panel.style.display = 'block';
    await loadAvailableModels();
  }
}

async function loadAvailableModels() {
  const listDiv = document.getElementById('model-list');
  listDiv.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 2rem;">Loading models...</p>';
  
  try {
    const res = await fetch(`${API}/models/browse?limit=100`);
    const data = await res.json();
    
    availableModels = data.models || [];
    document.getElementById('model-count').textContent = availableModels.length;
    
    renderModelList();
  } catch (e) {
    listDiv.innerHTML = `<p style="color: #ef4444; text-align: center; padding: 2rem;">Error loading models: ${e.message}</p>`;
  }
}

function filterModels() {
  const symbol = document.getElementById('filter-symbol').value.toLowerCase();
  const algorithm = document.getElementById('filter-algorithm').value.toLowerCase();
  const minAccuracy = parseFloat(document.getElementById('filter-accuracy').value) || 0;
  
  const filtered = availableModels.filter(m => {
    const matchSymbol = !symbol || m.symbol.toLowerCase().includes(symbol);
    const matchAlgo = !algorithm || m.algorithm.toLowerCase() === algorithm;
    const matchAcc = (m.accuracy * 100) >= minAccuracy;
    return matchSymbol && matchAlgo && matchAcc;
  });
  
  document.getElementById('model-count').textContent = filtered.length;
  renderModelList(filtered);
}

function renderModelList(models = null) {
  const listToRender = models || availableModels;
  const listDiv = document.getElementById('model-list');
  
  if (listToRender.length === 0) {
    listDiv.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 2rem;">No models found. Try adjusting filters or train some models first.</p>';
    return;
  }
  
  listDiv.innerHTML = listToRender.map(model => {
    const isSelected = selectedModelIds.has(model.id);
    const acc = model.accuracy ? (model.accuracy * 100).toFixed(1) : 'N/A';
    const r2 = model.r2_score ? (model.r2_score * 100).toFixed(1) : 'N/A';
    const mse = model.mse ? model.mse.toFixed(4) : 'N/A';
    
    return `
      <div style="background: ${isSelected ? 'rgba(16, 185, 129, 0.2)' : 'rgba(0,0,0,0.3)'}; 
                  border: 1px solid ${isSelected ? '#10b981' : 'rgba(255,255,255,0.1)'}; 
                  border-radius: 4px; 
                  padding: 0.75rem; 
                  cursor: pointer;"
           onclick="toggleModelSelection('${model.id}')">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
          <input type="checkbox" ${isSelected ? 'checked' : ''} onclick="event.stopPropagation(); toggleModelSelection('${model.id}')" style="cursor: pointer;">
          <div style="flex: 1;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
              <strong style="color: #60a5fa;">${model.symbol}</strong>
              <span style="background: rgba(59, 130, 246, 0.2); padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.75rem;">${model.algorithm}</span>
              <span style="font-size: 0.75rem; color: var(--text-muted);">${model.feature_count} features</span>
            </div>
            <div style="display: flex; gap: 1rem; font-size: 0.8rem;">
              <span>Acc: <strong style="color: ${acc >= 60 ? '#10b981' : '#ef4444'}">${acc}%</strong></span>
              <span>R²: <strong style="color: ${r2 >= 50 ? '#10b981' : '#ef4444'}">${r2}%</strong></span>
              <span>MSE: <strong>${mse}</strong></span>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">
              ID: ${model.id.substring(0, 12)}... | Transform: ${model.target_transform || 'none'}
            </div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function toggleModelSelection(modelId) {
  if (selectedModelIds.has(modelId)) {
    selectedModelIds.delete(modelId);
  } else {
    selectedModelIds.add(modelId);
  }
  updateSelectedModelsDisplay();
  renderModelList();
}

function selectAllModels() {
  const symbol = document.getElementById('filter-symbol').value.toLowerCase();
  const algorithm = document.getElementById('filter-algorithm').value.toLowerCase();
  const minAccuracy = parseFloat(document.getElementById('filter-accuracy').value) || 0;
  
  availableModels.forEach(m => {
    const matchSymbol = !symbol || m.symbol.toLowerCase().includes(symbol);
    const matchAlgo = !algorithm || m.algorithm.toLowerCase() === algorithm;
    const matchAcc = (m.accuracy * 100) >= minAccuracy;
    if (matchSymbol && matchAlgo && matchAcc) {
      selectedModelIds.add(m.id);
    }
  });
  
  updateSelectedModelsDisplay();
  renderModelList();
}

function clearAllModels() {
  selectedModelIds.clear();
  updateSelectedModelsDisplay();
  renderModelList();
}

function updateSelectedModelsDisplay() {
  const display = document.getElementById('selected-models-display');
  const countSpan = document.getElementById('selected-count');
  const btn = document.getElementById('run-selected-sims-btn');
  
  countSpan.textContent = selectedModelIds.size;
  btn.disabled = selectedModelIds.size === 0;
  
  if (selectedModelIds.size === 0) {
    display.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No models selected. Click "Browse Models" to select trained models for simulation.</span>';
  } else {
    const selectedModels = availableModels.filter(m => selectedModelIds.has(m.id));
    display.innerHTML = selectedModels.map(m => {
      const acc = m.accuracy ? (m.accuracy * 100).toFixed(1) : 'N/A';
      return `
        <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(16, 185, 129, 0.2); 
                    padding: 0.25rem 0.5rem; border-radius: 4px; margin: 0.25rem; border: 1px solid #10b981;">
          <span style="font-size: 0.85rem;"><strong>${m.symbol}</strong> ${m.algorithm} (${acc}%)</span>
          <button onclick="event.stopPropagation(); toggleModelSelection('${m.id}')" 
                  style="background: transparent; border: none; color: #ef4444; cursor: pointer; padding: 0; font-size: 1rem;">×</button>
        </div>
      `;
    }).join('');
  }
}

async function runSimulationsForSelectedModels() {
  if (selectedModelIds.size === 0) {
    alert('Please select at least one model first.');
    return;
  }
  
  const btn = document.getElementById('run-selected-sims-btn');
  const originalText = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = '⏳ Launching Simulations...';
  
  // Get simulation parameters
  const thresholds = document.getElementById('thresholds').value
    .split(',')
    .map(t => parseFloat(t.trim()))
    .filter(t => !isNaN(t));
  
  const zScores = document.getElementById('z-scores').value
    .split(',')
    .map(z => parseFloat(z.trim()))
    .filter(z => !isNaN(z));
  
  const regimeConfigs = getSelectedRegimeConfigs();
  
  // Get simulation tickers (or use model's training symbol)
  const simTickers = Array.from(selectedSimTickers);
  if (simTickers.length === 0) {
    // Use each model's training symbol
    const selectedModels = availableModels.filter(m => selectedModelIds.has(m.id));
    selectedModels.forEach(m => {
      if (!simTickers.includes(m.symbol)) {
        simTickers.push(m.symbol);
      }
    });
  }
  
  // Holy Grail criteria
  const sqnMin = parseFloat(document.getElementById('sqn-min')?.value || 3.0);
  const pfMin = parseFloat(document.getElementById('pf-min')?.value || 2.0);
  const tradesMin = parseInt(document.getElementById('trades-min')?.value || 200);
  
  const payload = {
    model_ids: Array.from(selectedModelIds),
    simulation_tickers: simTickers,
    thresholds: thresholds,
    z_score_thresholds: zScores,
    regime_configs: regimeConfigs,
    sqn_min: sqnMin,
    profit_factor_min: pfMin,
    trade_count_min: tradesMin
  };
  
  try {
    const res = await fetch(`${API}/simulations/manual`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    
    const resultDiv = document.getElementById('evolve-result');
    resultDiv.style.display = 'block';
    
    if (res.ok) {
      resultDiv.innerHTML = `
        <div class="alert success">
          ✅ Simulations launched!<br>
          Models: ${data.model_count}<br>
          Tickers: ${data.ticker_count}<br>
          Simulations per model: ${data.simulations_per_model}<br>
          <strong>Total simulations queued: ${data.total_simulations}</strong><br>
          Jobs queued: ${data.total_jobs}
        </div>
      `;
      loadRuns();
      loadStats();
    } else {
      resultDiv.innerHTML = `
        <div class="alert error">
          ❌ Error: ${data.detail || JSON.stringify(data)}
        </div>
      `;
    }
  } catch (e) {
    document.getElementById('evolve-result').innerHTML = `
      <div class="alert error">
        ❌ Network error: ${e.message}
      </div>
    `;
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalText;
  }
}

// ============================================
// Train Models Only (Skip Simulations)
// ============================================

// ============================================
// Helper: Collect Hyperparameter Grids
// ============================================

function collectHyperparameterGrids() {
  const algorithm = document.getElementById('algorithm')?.value || '';
  const algoLower = algorithm.toLowerCase();
  
  const grids = {};
  
  // ElasticNet/Ridge/Lasso grids
  if (algoLower.includes('elasticnet') || algoLower.includes('ridge') || algoLower.includes('lasso')) {
    const alphaGridStr = document.getElementById('alpha-grid')?.value || '';
    const l1RatioGridStr = document.getElementById('l1-ratio-grid')?.value || '';
    const alphaGrid = alphaGridStr.split(',').map(a => parseFloat(a.trim())).filter(a => !isNaN(a));
    const l1RatioGrid = l1RatioGridStr.split(',').map(l => parseFloat(l.trim())).filter(l => !isNaN(l));
    
    if (alphaGrid.length > 0) grids.alpha_grid = alphaGrid;
    if (l1RatioGrid.length > 0) grids.l1_ratio_grid = l1RatioGrid;
  }
  
  // XGBoost grids
  else if (algoLower.includes('xgboost')) {
    const maxDepthStr = document.getElementById('max-depth-grid')?.value || '';
    const minChildWeightStr = document.getElementById('min-child-weight-grid')?.value || '';
    const regLambdaStr = document.getElementById('reg-lambda-grid')?.value || '';
    const learningRateStr = document.getElementById('learning-rate-grid')?.value || '';
    
    const maxDepthGrid = maxDepthStr.split(',').map(d => parseInt(d.trim())).filter(d => !isNaN(d));
    const minChildWeightGrid = minChildWeightStr.split(',').map(w => parseInt(w.trim())).filter(w => !isNaN(w));
    const regLambdaGrid = regLambdaStr.split(',').map(l => parseFloat(l.trim())).filter(l => !isNaN(l));
    const learningRateGrid = learningRateStr.split(',').map(r => parseFloat(r.trim())).filter(r => !isNaN(r));
    
    if (maxDepthGrid.length > 0) grids.max_depth_grid = maxDepthGrid;
    if (minChildWeightGrid.length > 0) grids.min_child_weight_grid = minChildWeightGrid;
    if (regLambdaGrid.length > 0) grids.reg_lambda_grid = regLambdaGrid;
    if (learningRateGrid.length > 0) grids.learning_rate_grid = learningRateGrid;
  }
  
  // LightGBM grids
  else if (algoLower.includes('lightgbm')) {
    const numLeavesStr = document.getElementById('num-leaves-grid')?.value || '';
    const minDataInLeafStr = document.getElementById('min-data-in-leaf-grid')?.value || '';
    const lambdaL2Str = document.getElementById('lambda-l2-grid')?.value || '';
    const learningRateStr = document.getElementById('lgbm-learning-rate-grid')?.value || '';
    
    const numLeavesGrid = numLeavesStr.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n));
    const minDataInLeafGrid = minDataInLeafStr.split(',').map(d => parseInt(d.trim())).filter(d => !isNaN(d));
    const lambdaL2Grid = lambdaL2Str.split(',').map(l => parseFloat(l.trim())).filter(l => !isNaN(l));
    const learningRateGrid = learningRateStr.split(',').map(r => parseFloat(r.trim())).filter(r => !isNaN(r));
    
    if (numLeavesGrid.length > 0) grids.num_leaves_grid = numLeavesGrid;
    if (minDataInLeafGrid.length > 0) grids.min_data_in_leaf_grid = minDataInLeafGrid;
    if (lambdaL2Grid.length > 0) grids.lambda_l2_grid = lambdaL2Grid;
    if (learningRateGrid.length > 0) grids.lgbm_learning_rate_grid = learningRateGrid;
  }
  
  // Random Forest grids
  else if (algoLower.includes('random_forest') || algoLower.includes('randomforest')) {
    const maxDepthStr = document.getElementById('rf-max-depth-grid')?.value || '';
    const minSamplesSplitStr = document.getElementById('min-samples-split-grid')?.value || '';
    const minSamplesLeafStr = document.getElementById('min-samples-leaf-grid')?.value || '';
    const nEstimatorsStr = document.getElementById('n-estimators-grid')?.value || '';
    
    // Parse max_depth (handle 'None' as string)
    const maxDepthGrid = maxDepthStr.split(',').map(d => {
      const trimmed = d.trim();
      if (trimmed.toLowerCase() === 'none') return null;
      const parsed = parseInt(trimmed);
      return isNaN(parsed) ? null : parsed;
    });
    
    const minSamplesSplitGrid = minSamplesSplitStr.split(',').map(s => parseInt(s.trim())).filter(s => !isNaN(s));
    const minSamplesLeafGrid = minSamplesLeafStr.split(',').map(l => parseInt(l.trim())).filter(l => !isNaN(l));
    const nEstimatorsGrid = nEstimatorsStr.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n));
    
    if (maxDepthGrid.length > 0) grids.rf_max_depth_grid = maxDepthGrid;
    if (minSamplesSplitGrid.length > 0) grids.min_samples_split_grid = minSamplesSplitGrid;
    if (minSamplesLeafGrid.length > 0) grids.min_samples_leaf_grid = minSamplesLeafGrid;
    if (nEstimatorsGrid.length > 0) grids.n_estimators_grid = nEstimatorsGrid;
  }
  
  return grids;
}

// ============================================
// Train Models Only (Skip Simulations)
// ============================================

async function trainModelsOnly() {
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = '⏳ Starting Training...';
  
  // Collect all hyperparameter grids based on selected algorithm
  const hyperparamGrids = collectHyperparameterGrids();
  
  const payload = {
    seed_model_id: document.getElementById('seed-model-id').value || null,
    symbol: document.getElementById('symbol').value,
    reference_symbols: Array.from(selectedReferences),  // Include reference symbols
    data_options: selectedDataOptions,  // Include selected fold/options
    algorithm: document.getElementById('algorithm').value,
    target_col: document.getElementById('target-col').value,
    max_generations: parseInt(document.getElementById('max-generations').value),
    prune_fraction: parseInt(document.getElementById('prune-fraction').value) / 100,  // Convert % to decimal
    min_features: parseInt(document.getElementById('min-features').value),
    target_transform: document.getElementById('target-transform').value,
    timeframe: document.getElementById('timeframe').value,
    ...hyperparamGrids  // Spread all collected grids into payload
  };
  
  try {
    const res = await fetch(`${API}/train-only`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    
    const resultDiv = document.getElementById('evolve-result');
    resultDiv.style.display = 'block';
    
    if (res.ok) {
      resultDiv.innerHTML = `
        <div class="alert success">
          ✅ Training started! Run ID: ${data.run_id?.substring(0, 12)}...<br>
          Target: ${payload.symbol}${payload.reference_symbols.length > 0 ? ` + ${payload.reference_symbols.length} references` : ''}<br>
          <strong>Training only - no simulations will run.</strong><br>
          After training completes, visit <a href="/models" style="color: #60a5fa; text-decoration: underline;">Model Browser</a> to review and select models for simulation.
        </div>
      `;
      loadRuns();
      loadStats();
    } else {
      resultDiv.innerHTML = `
        <div class="alert error">
          ❌ Error: ${data.detail || JSON.stringify(data)}
        </div>
      `;
    }
  } catch (e) {
    resultDiv.innerHTML = `
      <div class="alert error">
        ❌ Network error: ${e.message}
      </div>
    `;
  } finally {
    btn.disabled = false;
    btn.textContent = '🧠 Train Models Only (Skip Simulations)';
  }
}

// ============================================
// Pure Grid Search Functions (No Evolution/Pruning)
// ============================================

async function gridSearchElasticNet(btn) {
  console.log('gridSearchElasticNet called');
  const symbol = document.getElementById('symbol').value.trim();
  console.log('Symbol:', symbol);
  
  if (!symbol) {
    alert('Please enter a ticker symbol');
    return;
  }
  
  if (!btn) btn = event.target;
  console.log('Button:', btn);
  
  btn.disabled = true;
  btn.textContent = '⏳ Training...';
  
  try {
    const payload = {
      symbol: symbol,
      reference_symbols: Array.from(selectedReferences),
      target_col: document.getElementById('target-col').value,
      target_transform: document.getElementById('target-transform').value,
      timeframe: document.getElementById('timeframe').value,
      alpha_grid: document.getElementById('alpha-grid').value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
      l1_ratio_grid: document.getElementById('l1-ratio-grid').value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
    };
    
    // Check if multi-generational pruning is enabled
    const enablePruning = document.getElementById('grid-enable-pruning')?.checked || false;
    if (enablePruning) {
      payload.max_generations = parseInt(document.getElementById('max-generations').value) || 4;
      payload.prune_fraction = parseFloat(document.getElementById('prune-fraction').value) / 100 || 0.25;
      payload.min_features = parseInt(document.getElementById('min-features').value) || 5;
    }
    
    console.log('ElasticNet payload:', payload);
    
    const gridSize = payload.alpha_grid.length * payload.l1_ratio_grid.length;
    console.log('Grid size:', gridSize);
    
    let confirmMsg = `This will train ${gridSize} ElasticNet models`;
    if (enablePruning) {
      confirmMsg += ` across up to ${payload.max_generations} generations with feature pruning`;
    }
    confirmMsg += '. Continue?';
    
    if (!confirm(confirmMsg)) {
      btn.disabled = false;
      btn.textContent = '🔹 ElasticNet Grid';
      console.log('User cancelled');
      return;
    }
    
    console.log('Sending request to /grid-search/elasticnet');
    const res = await fetch('/grid-search/elasticnet', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    console.log('Response status:', res.status);
    const data = await res.json();
    console.log('Response data:', data);
    
    const resultDiv = document.getElementById('evolve-result');
    resultDiv.style.display = 'block';
    
    if (res.ok) {
      resultDiv.innerHTML = `
        <div class="alert success">
          ✅ ElasticNet Grid Search Started!<br>
          Training ${data.grid_size} models for ${symbol} in background<br>
          Run ID: ${data.run_id}<br>
          Check "Active Runs" tab to monitor progress
        </div>
      `;
      loadStats();
      loadRuns();  // Refresh runs to show new grid search
    } else {
      resultDiv.innerHTML = `<div class="alert error">❌ Error: ${data.detail || JSON.stringify(data)}</div>`;
    }
  } catch (e) {
    document.getElementById('evolve-result').innerHTML = `<div class="alert error">❌ Network error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🔹 ElasticNet Grid';
  }
}

async function gridSearchXGBoost(btn) {
  const symbol = document.getElementById('symbol').value.trim();
  if (!symbol) {
    alert('Please enter a ticker symbol');
    return;
  }
  
  if (!btn) btn = event.target;
  btn.disabled = true;
  btn.textContent = '⏳ Training...';
  
  try {
    const payload = {
      symbol: symbol,
      reference_symbols: Array.from(selectedReferences),
      target_col: document.getElementById('target-col').value,
      target_transform: document.getElementById('target-transform').value,
      timeframe: document.getElementById('timeframe').value,
      regressor: true,
      max_depth_grid: document.getElementById('max-depth-grid').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)),
      min_child_weight_grid: document.getElementById('min-child-weight-grid').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)),
      reg_alpha_grid: document.getElementById('reg-alpha-grid').value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
      reg_lambda_grid: document.getElementById('reg-lambda-grid').value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
      learning_rate_grid: document.getElementById('learning-rate-grid').value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
    };
    
    // Check if multi-generational pruning is enabled
    const enablePruning = document.getElementById('grid-enable-pruning')?.checked || false;
    if (enablePruning) {
      payload.max_generations = parseInt(document.getElementById('max-generations').value) || 4;
      payload.prune_fraction = parseFloat(document.getElementById('prune-fraction').value) / 100 || 0.25;
      payload.min_features = parseInt(document.getElementById('min-features').value) || 5;
    }
    
    const gridSize = payload.max_depth_grid.length * payload.min_child_weight_grid.length * 
                     payload.reg_alpha_grid.length * payload.reg_lambda_grid.length * payload.learning_rate_grid.length;
    
    let confirmMsg = `This will train ${gridSize} XGBoost models (may take 15-30 mins)`;
    if (enablePruning) {
      confirmMsg += ` across up to ${payload.max_generations} generations with feature pruning`;
    }
    confirmMsg += '. Continue?';
    
    if (!confirm(confirmMsg)) {
      btn.disabled = false;
      btn.textContent = '🌲 XGBoost Grid';
      return;
    }
    
    const res = await fetch('/grid-search/xgboost', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    
    const resultDiv = document.getElementById('evolve-result');
    resultDiv.style.display = 'block';
    
    if (res.ok) {
      resultDiv.innerHTML = `
        <div class="alert success">
          ✅ XGBoost Grid Search Started!<br>
          Training ${data.grid_size} models for ${symbol} in background<br>
          Run ID: ${data.run_id}<br>
          Check "Active Runs" tab to monitor progress
        </div>
      `;
      loadStats();
      loadRuns();  // Refresh active runs
    } else {
      resultDiv.innerHTML = `<div class="alert error">❌ Error: ${data.detail || JSON.stringify(data)}</div>`;
    }
  } catch (e) {
    document.getElementById('evolve-result').innerHTML = `<div class="alert error">❌ Network error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🌲 XGBoost Grid';
  }
}

async function gridSearchLightGBM(btn) {
  const symbol = document.getElementById('symbol').value.trim();
  if (!symbol) {
    alert('Please enter a ticker symbol');
    return;
  }
  
  if (!btn) btn = event.target;
  btn.disabled = true;
  btn.textContent = '⏳ Training...';
  
  try {
    const payload = {
      symbol: symbol,
      reference_symbols: Array.from(selectedReferences),
      target_col: document.getElementById('target-col').value,
      target_transform: document.getElementById('target-transform').value,
      timeframe: document.getElementById('timeframe').value,
      regressor: true,
      num_leaves_grid: document.getElementById('num-leaves-grid').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)),
      min_data_in_leaf_grid: document.getElementById('min-data-in-leaf-grid').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)),
      lambda_l1_grid: document.getElementById('lambda-l1-grid').value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
      lambda_l2_grid: document.getElementById('lambda-l2-grid').value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v)),
      learning_rate_grid: document.getElementById('lgbm-learning-rate-grid').value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
    };
    
    // Check if multi-generational pruning is enabled
    const enablePruning = document.getElementById('grid-enable-pruning')?.checked || false;
    if (enablePruning) {
      payload.max_generations = parseInt(document.getElementById('max-generations').value) || 4;
      payload.prune_fraction = parseFloat(document.getElementById('prune-fraction').value) / 100 || 0.25;
      payload.min_features = parseInt(document.getElementById('min-features').value) || 5;
    }
    
    const gridSize = payload.num_leaves_grid.length * payload.min_data_in_leaf_grid.length * 
                     payload.lambda_l1_grid.length * payload.lambda_l2_grid.length * payload.learning_rate_grid.length;
    
    let confirmMsg = `This will train ${gridSize} LightGBM models (may take 15-30 mins)`;
    if (enablePruning) {
      confirmMsg += ` across up to ${payload.max_generations} generations with feature pruning`;
    }
    confirmMsg += '. Continue?';
    
    if (!confirm(confirmMsg)) {
      btn.disabled = false;
      btn.textContent = '💡 LightGBM Grid';
      return;
    }
    
    const res = await fetch('/grid-search/lightgbm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    
    const resultDiv = document.getElementById('evolve-result');
    resultDiv.style.display = 'block';
    
    if (res.ok) {
      resultDiv.innerHTML = `
        <div class="alert success">
          ✅ LightGBM Grid Search Started!<br>
          Training ${data.grid_size} models for ${symbol} in background<br>
          Run ID: ${data.run_id}<br>
          Check "Active Runs" tab to monitor progress
        </div>
      `;
      loadStats();
      loadRuns();  // Refresh active runs
    } else {
      resultDiv.innerHTML = `<div class="alert error">❌ Error: ${data.detail || JSON.stringify(data)}</div>`;
    }
  } catch (e) {
    document.getElementById('evolve-result').innerHTML = `<div class="alert error">❌ Network error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '💡 LightGBM Grid';
  }
}

async function gridSearchRandomForest(btn) {
  const symbol = document.getElementById('symbol').value.trim();
  if (!symbol) {
    alert('Please enter a ticker symbol');
    return;
  }
  
  if (!btn) btn = event.target;
  btn.disabled = true;
  btn.textContent = '⏳ Training...';
  
  try {
    // Parse max_depth_grid - handle "None" as null
    const maxDepthRaw = document.getElementById('rf-max-depth-grid').value.split(',').map(v => v.trim());
    const maxDepthGrid = maxDepthRaw.map(v => {
      if (v.toLowerCase() === 'none') return null;
      const num = parseInt(v);
      return isNaN(num) ? null : num;
    });
    
    // Parse max_features_grid - handle "sqrt", "log2", and numeric values
    const maxFeaturesRaw = document.getElementById('max-features-grid').value.split(',').map(v => v.trim());
    const maxFeaturesGrid = maxFeaturesRaw.map(v => {
      if (v.toLowerCase() === 'sqrt' || v.toLowerCase() === 'log2') return v.toLowerCase();
      const num = parseFloat(v);
      return isNaN(num) ? v : num;
    });
    
    const payload = {
      symbol: symbol,
      reference_symbols: Array.from(selectedReferences),
      target_col: document.getElementById('target-col').value,
      target_transform: document.getElementById('target-transform').value,
      timeframe: document.getElementById('timeframe').value,
      regressor: true,
      max_depth_grid: maxDepthGrid,
      min_samples_split_grid: document.getElementById('min-samples-split-grid').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)),
      min_samples_leaf_grid: document.getElementById('min-samples-leaf-grid').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)),
      n_estimators_grid: document.getElementById('n-estimators-grid').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)),
      max_features_grid: maxFeaturesGrid
    };
    
    // Check if multi-generational pruning is enabled
    const enablePruning = document.getElementById('grid-enable-pruning')?.checked || false;
    if (enablePruning) {
      payload.max_generations = parseInt(document.getElementById('max-generations').value) || 4;
      payload.prune_fraction = parseFloat(document.getElementById('prune-fraction').value) / 100 || 0.25;
      payload.min_features = parseInt(document.getElementById('min-features').value) || 5;
    }
    
    const gridSize = payload.max_depth_grid.length * payload.min_samples_split_grid.length * 
                     payload.min_samples_leaf_grid.length * payload.n_estimators_grid.length * payload.max_features_grid.length;
    
    let confirmMsg = `This will train ${gridSize} RandomForest models (may take 15-30 mins)`;
    if (enablePruning) {
      confirmMsg += ` across up to ${payload.max_generations} generations with feature pruning`;
    }
    confirmMsg += '. Continue?';
    
    if (!confirm(confirmMsg)) {
      btn.disabled = false;
      btn.textContent = '🌳 RandomForest Grid';
      return;
    }
    
    const res = await fetch('/grid-search/randomforest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    
    const resultDiv = document.getElementById('evolve-result');
    resultDiv.style.display = 'block';
    
    if (res.ok) {
      resultDiv.innerHTML = `
        <div class="alert success">
          ✅ RandomForest Grid Search Started!<br>
          Training ${data.grid_size} models for ${symbol} in background<br>
          Run ID: ${data.run_id}<br>
          Check "Active Runs" tab to monitor progress
        </div>
      `;
      loadStats();
      loadRuns();  // Refresh active runs
    } else {
      resultDiv.innerHTML = `<div class="alert error">❌ Error: ${data.detail || JSON.stringify(data)}</div>`;
    }
  } catch (e) {
    document.getElementById('evolve-result').innerHTML = `<div class="alert error">❌ Network error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🌳 RandomForest Grid';
  }
}

// ============================================
// Form Handler (Full Evolution)
// ============================================

function setupFormHandler() {
  const form = document.getElementById('evolve-form');
  if (!form) return;
  
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = e.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.textContent = '⏳ Starting...';
    
    const thresholds = document.getElementById('thresholds').value
      .split(',')
      .map(t => parseFloat(t.trim()))
      .filter(t => !isNaN(t));
    
    const zScores = document.getElementById('z-scores').value
      .split(',')
      .map(z => parseFloat(z.trim()))
      .filter(z => !isNaN(z));
    
    // Regime configs - dynamically built from checkboxes
    const regimeConfigs = getSelectedRegimeConfigs();
    
    // Collect all hyperparameter grids based on selected algorithm
    const hyperparamGrids = collectHyperparameterGrids();
    
    const payload = {
      seed_model_id: document.getElementById('seed-model-id').value || null,
      symbol: document.getElementById('symbol').value,
      reference_symbols: Array.from(selectedReferences),  // Include reference symbols
      simulation_tickers: selectedSimTickers.size > 0 ? Array.from(selectedSimTickers) : null,  // Simulation tickers (null = use training symbol)
      data_options: selectedDataOptions,  // Include selected fold/options
      algorithm: document.getElementById('algorithm').value,
      target_col: document.getElementById('target-col').value,
      max_generations: parseInt(document.getElementById('max-generations').value),
      prune_fraction: parseInt(document.getElementById('prune-fraction').value) / 100,  // Convert % to decimal
      min_features: parseInt(document.getElementById('min-features').value),
      target_transform: document.getElementById('target-transform').value,
      timeframe: document.getElementById('timeframe').value,
      thresholds: thresholds,
      z_score_thresholds: zScores,
      regime_configs: regimeConfigs,
      ...hyperparamGrids,  // Spread all collected grids into payload
      sqn_min: parseFloat(document.getElementById('sqn-min').value),
      sqn_max: parseFloat(document.getElementById('sqn-max').value),
      profit_factor_min: parseFloat(document.getElementById('pf-min').value),
      profit_factor_max: parseFloat(document.getElementById('pf-max').value),
      trade_count_min: parseInt(document.getElementById('trades-min').value),
      trade_count_max: parseInt(document.getElementById('trades-max').value)
    };
    
    try {
      const res = await fetch(`${API}/evolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      const data = await res.json();
      
      const resultDiv = document.getElementById('evolve-result');
      resultDiv.style.display = 'block';
      
      if (res.ok) {
        resultDiv.innerHTML = `
          <div class="alert success">
            ✅ Evolution started! Run ID: ${data.run_id?.substring(0, 12)}...<br>
            Target: ${payload.symbol}${payload.reference_symbols.length > 0 ? ` + ${payload.reference_symbols.length} references` : ''}<br>
            Monitoring in Active Runs section.
          </div>
        `;
        loadRuns();
        loadStats();
      } else {
        resultDiv.innerHTML = `
          <div class="alert error">
            ❌ Error: ${data.detail || JSON.stringify(data)}
          </div>
        `;
      }
    } catch (e) {
      document.getElementById('evolve-result').innerHTML = `
        <div class="alert error">
          ❌ Network error: ${e.message}
        </div>
      `;
    } finally {
      btn.disabled = false;
      btn.textContent = '🧬 Start Evolution';
    }
  });
}
// ============================================
// Live Logs Terminal
// ============================================

function toggleLogsHeight() {
  const container = document.getElementById('logs-container');
  const btn = document.getElementById('logs-toggle-btn');
  const isExpanded = container.getAttribute('data-expanded') === 'true';
  
  if (isExpanded) {
    // Collapse
    container.style.height = '100px';
    container.setAttribute('data-expanded', 'false');
    btn.textContent = '⬆️ Expand';
  } else {
    // Expand
    container.style.height = '1000px';
    container.setAttribute('data-expanded', 'true');
    btn.textContent = '⬇️ Collapse';
  }
}

async function loadLogs() {
  try {
    // Fetch logs from all three services via orchestrator proxy
    const [orchResp, trainResp, simResp] = await Promise.all([
      fetch(`${API}/logs`).catch(() => ({json: () => []})),
      fetch(`${API}/training/logs`).catch(() => ({json: () => []})),
      fetch(`${API}/simulation/logs`).catch(() => ({json: () => []}))
    ]);
    
    const orchLogs = await orchResp.json();
    const trainLogs = await trainResp.json();
    const simLogs = await simResp.json();
    
    // Combine and tag logs by service
    const allLogs = [
      ...orchLogs.map(l => ({text: l, service: 'ORCH'})),
      ...trainLogs.map(l => ({text: l, service: 'TRAIN'})),
      ...simLogs.map(l => ({text: l, service: 'SIM'}))
    ];
    
    // Sort by timestamp (extract from log line)
    allLogs.sort((a, b) => {
      const timeA = a.text.substring(0, 19); // YYYY-MM-DD HH:MM:SS
      const timeB = b.text.substring(0, 19);
      return timeA.localeCompare(timeB);
    });
    
    const container = document.getElementById('logs-container');
    if (!container) return;
    
    const autoScroll = document.getElementById('auto-scroll-logs')?.checked ?? true;
    const wasAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;
    
    // Format logs with color coding and service badges
    container.innerHTML = allLogs.map(({text, service}) => {
      let color = '#e5e7eb'; // default gray
      if (text.includes('ERROR')) color = '#ef4444';
      else if (text.includes('WARNING')) color = '#f59e0b';
      else if (text.includes('INFO')) color = '#10b981';
      else if (text.includes('DEBUG')) color = '#6b7280';
      
      // Service badge colors
      let badgeColor = '#3b82f6'; // blue for ORCH
      if (service === 'TRAIN') badgeColor = '#8b5cf6'; // purple
      if (service === 'SIM') badgeColor = '#ec4899'; // pink
      
      return `<div style="color: ${color};"><span style="background: ${badgeColor}; padding: 1px 4px; border-radius: 3px; font-size: 0.7rem; margin-right: 6px;">${service}</span>${escapeHtml(text)}</div>`;
    }).join('');
    
    // Auto-scroll if enabled and was at bottom
    if (autoScroll && wasAtBottom) {
      container.scrollTop = container.scrollHeight;
    }
  } catch (e) {
    console.error('Failed to load logs:', e);
  }
}

function clearLogs() {
  const container = document.getElementById('logs-container');
  if (container) {
    container.innerHTML = '<div style="color: #6b7280;">Logs cleared. Waiting for new entries...</div>';
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
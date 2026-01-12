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
}, 10000);

// Initial load
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadRuns();
  loadPromoted();
  setupFormHandler();
  setupSymbolSelector();
});

// ============================================
// Symbol Selection from Feature Service
// ============================================

let availableOptions = [];
let availableSymbols = [];
let selectedReferences = new Set();
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
      
      // Update reference checkboxes
      updateReferenceCheckboxes();
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
    targetSelect.addEventListener('change', updateReferenceCheckboxes);
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
      tbody.innerHTML = '<tr><td colspan="7" style="color: var(--text-muted);">No evolution runs yet. Start one above!</td></tr>';
      return;
    }
    
    tbody.innerHTML = data.runs.map(run => `
      <tr class="clickable" onclick="showDetails('${run.id}')">
        <td><code>${run.id.substring(0, 8)}...</code></td>
        <td>${run.symbol}</td>
        <td><span class="badge ${run.status.toLowerCase()}">${run.status}</span></td>
        <td>${run.current_generation} / ${run.max_generations}</td>
        <td>${run.best_sqn ? run.best_sqn.toFixed(2) : '-'}</td>
        <td>${new Date(run.created_at).toLocaleString()}</td>
        <td><button class="secondary" onclick="event.stopPropagation(); showDetails('${run.id}')">View</button></td>
      </tr>
    `).join('');
    
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
  return `
    <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 6px; margin-bottom: 0.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <strong>${run.symbol}</strong>
          <span class="badge ${run.status.toLowerCase()}" style="margin-left: 0.5rem;">${run.status}</span>
        </div>
        <code style="font-size: 0.8rem;">${run.id.substring(0, 8)}</code>
      </div>
      <div style="margin-top: 0.5rem;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted);">
          <span>Generation ${run.current_generation} of ${run.max_generations}</span>
          <span>Best SQN: ${run.best_sqn ? run.best_sqn.toFixed(2) : '-'}</span>
        </div>
        <div class="progress-bar" style="margin-top: 0.25rem;">
          <div class="progress-fill" style="width: ${(run.current_generation / run.max_generations) * 100}%"></div>
        </div>
      </div>
    </div>
  `;
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
    
    const payload = {
      seed_model_id: document.getElementById('seed-model-id').value || null,
      symbol: document.getElementById('symbol').value,
      reference_symbols: Array.from(selectedReferences),  // Include reference symbols
      data_options: selectedDataOptions,  // Include selected fold/options
      algorithm: document.getElementById('algorithm').value,
      target_col: document.getElementById('target-col').value,
      max_generations: parseInt(document.getElementById('max-generations').value),
      target_transform: document.getElementById('target-transform').value,
      timeframe: document.getElementById('timeframe').value,
      thresholds: thresholds,
      regime_configs: [
        {"regime_gmm": [0]},
        {"regime_gmm": [1]},
        {"regime_vix": [0, 1]}
      ],
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

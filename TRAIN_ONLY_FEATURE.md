# Train Models Only Feature

## Overview
Added a new "Train Models Only" button to the Training Configuration section that allows training and pruning models WITHOUT running simulations. This enables a two-phase workflow for better control and visibility.

## Changes Made

### 1. Dashboard UI (`orchestrator_service/templates/dashboard_new.html`)

**Added Train Models Only Button** (after regularization grid, ~line 207):
- Blue highlighted panel with clear description
- Button text: "🧠 Train Models Only (Skip Simulations)"
- Explanation: Grid search models for X generations WITHOUT running simulations
- Positioned BEFORE the simulation configuration section

**Updated Full Evolution Button** (line 338):
- Changed text from "🧬 Start Evolution" to "🧬 Start Full Evolution (Train + Simulate)"
- Changed background color to green (#16a34a) to differentiate from train-only (blue)
- Makes it clear this runs the complete pipeline

### 2. JavaScript (`orchestrator_service/static/js/dashboard.js`)

**Added `trainModelsOnly()` function** (~line 850):
- Calls `/train-only` endpoint (POST)
- Extracts training configuration from form:
  - seed_model_id
  - symbol
  - reference_symbols
  - data_options
  - algorithm
  - target_col
  - max_generations
  - prune_fraction
  - min_features
  - target_transform
  - timeframe
  - alpha_grid (for ElasticNet/Ridge/Lasso)
  - l1_ratio_grid
- Skips all simulation parameters (thresholds, z-scores, regime_configs, Holy Grail criteria)
- Shows success message with link to Model Browser (`/models`)
- Updates runs and stats displays

## Workflow

### Phase 1: Train Models
1. Configure training parameters in Training Configuration section
2. Click "🧠 Train Models Only (Skip Simulations)"
3. System trains models for X generations
4. Models are pruned based on training metrics (accuracy, R², MSE)
5. NO simulations are run

### Phase 2: Manual Review & Selection
1. Visit `/models` page (Model Browser)
2. Filter by symbol, algorithm, status, accuracy
3. Review model fingerprints:
   - Hyperparameters
   - Feature counts
   - Training metrics
   - Feature importance
4. Select models via checkboxes

### Phase 3: Manual Simulation Launch
1. Configure simulation parameters in Model Browser
2. Click "Launch Grid Search Simulations"
3. Only selected models run simulations
4. Grid search across thresholds, z-scores, regimes

## Benefits

1. **Better Control**: Separate training from simulation stages
2. **Visibility**: Review models before committing to expensive simulations
3. **Efficiency**: Only simulate promising models
4. **Experimentation**: Test different training configs quickly without simulation overhead
5. **Debugging**: Isolate training issues from simulation issues

## Testing

To test the feature:

1. **Restart orchestrator service** (if needed):
   ```bash
   docker-compose restart orchestrator
   ```

2. **Access dashboard**:
   ```bash
   $BROWSER http://localhost:8000
   ```

3. **Test train-only workflow**:
   - Fill in training configuration (symbol, algorithm, generations, etc.)
   - Click "🧠 Train Models Only (Skip Simulations)"
   - Verify run starts and appears in Active Runs
   - Check logs to confirm no simulations are running

4. **Verify full evolution still works**:
   - Fill in both training AND simulation configuration
   - Click "🧬 Start Full Evolution (Train + Simulate)"
   - Verify both training and simulations run

5. **Test model browser integration**:
   - After training completes, visit http://localhost:8000/models
   - Verify models appear with correct fingerprints
   - Test manual simulation launcher

## API Endpoint

The new button calls the existing `/train-only` endpoint:

```
POST /train-only
Content-Type: application/json

{
  "seed_model_id": "...",
  "symbol": "AAPL",
  "reference_symbols": ["SPY", "QQQ"],
  "data_options": [...],
  "algorithm": "randomforest",
  "target_col": "close_zscore_change",
  "max_generations": 5,
  "prune_fraction": 0.5,
  "min_features": 5,
  "target_transform": "log",
  "timeframe": "1d",
  "alpha_grid": [0.0001, 0.001, 0.01, 0.1],
  "l1_ratio_grid": [0.1, 0.5, 0.9]
}
```

Response:
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Training started for AAPL"
}
```

## Files Modified

1. `orchestrator_service/templates/dashboard_new.html`
   - Added Train Models Only button and panel
   - Updated Full Evolution button text and color

2. `orchestrator_service/static/js/dashboard.js`
   - Added `trainModelsOnly()` async function
   - Wired up button to call `/train-only` endpoint

## Notes

- Train-only mode uses ALL training configuration fields
- Train-only mode SKIPS all simulation configuration fields
- Both buttons are now clearly labeled with their behavior
- Success message includes link to Model Browser for next steps
- Full evolution workflow remains unchanged for users who want everything automated

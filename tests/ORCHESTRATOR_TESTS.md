# Orchestrator Workflow Tests

This directory contains comprehensive test coverage for the new two-phase workflow:

1. **Train Models Only** - Train and prune models without simulations
2. **Browse & Select Models** - Filter and select trained models
3. **Run Simulations** - Launch simulations for selected models

## Test Files

### Unit Tests: `tests/unit/test_orchestrator_ui.py`

Tests pure Python logic for UI interactions (no API calls).

**Classes:**
- `TestModelBrowserJavaScript` - Model selection logic
  - `test_toggle_model_selection()` - Add/remove models from selection
  - `test_select_all_models_with_filter()` - Select all respects filters
  - `test_grid_size_calculation()` - Simulation count math
  - `test_update_selected_models_display()` - Display rendering logic

- `TestSimulationLauncher` - Simulation launching
  - `test_manual_simulation_payload()` - Payload construction
  - `test_button_disabled_when_no_models_selected()` - Button state
  - `test_simulation_ticker_fallback()` - Ticker selection logic

- `TestTrainOnlyButton` - Train-only mode
  - `test_train_only_payload()` - Payload excludes simulation params

- `TestFilterModels` - Model filtering
  - `test_filter_by_symbol()` - Symbol partial match
  - `test_filter_by_algorithm()` - Algorithm exact match
  - `test_filter_by_min_accuracy()` - Accuracy threshold
  - `test_combined_filters()` - Multiple filters simultaneously

**Run:**
```bash
python -m pytest tests/unit/test_orchestrator_ui.py -v
```

### Integration Tests: `tests/integration/test_orchestrator_workflows.py`

Tests API endpoints with live services (requires docker-compose).

**Classes:**
- `TestTrainOnlyWorkflow` - End-to-end train-only
  - `test_train_only_creates_models_no_simulations()` - Full workflow
  - `test_train_only_with_regularization_grid()` - ElasticNet grid search

- `TestModelBrowseAndSelect` - Model browsing API
  - `test_browse_models_with_filters()` - Filter parameters
  - `test_model_fingerprint_included()` - Response structure

- `TestManualSimulations` - Manual simulation launcher
  - `test_run_simulations_for_selected_models()` - Full workflow
  - `test_manual_simulations_validation()` - Input validation

- `TestFullEvolutionComparison` - Backwards compatibility
  - `test_full_evolution_creates_models_and_simulations()` - Traditional flow

**Setup:**
```bash
# Start services
docker-compose up -d

# Run tests
python -m pytest tests/integration/test_orchestrator_workflows.py -v -m integration
```

## Test Coverage

### Workflow 1: Train Models Only

**Endpoints Tested:**
- `POST /train-only` - Submit training request

**Validations:**
- ✅ Response indicates train-only mode
- ✅ Models created in database
- ✅ No simulations run
- ✅ Regularization grid creates multiple models
- ✅ Feature columns auto-fetched from parquet

**Test Scenarios:**
1. Basic train-only with RandomForest
2. Train-only with ElasticNet + alpha/l1_ratio grid
3. With reference symbols
4. Without seed features (auto-fetch)

### Workflow 2: Browse & Select Models

**Endpoints Tested:**
- `GET /models/browse` - List models with filters

**Validations:**
- ✅ Symbol filter (partial match)
- ✅ Algorithm filter (exact match)
- ✅ Status filter
- ✅ Minimum accuracy filter
- ✅ Limit parameter
- ✅ Response includes fingerprint data
- ✅ Metrics included (accuracy, R², MSE)

**Test Scenarios:**
1. Browse all models
2. Filter by single parameter
3. Combine multiple filters
4. Pagination with limit

### Workflow 3: Run Simulations for Selected Models

**Endpoints Tested:**
- `POST /simulations/manual` - Launch simulations

**Validations:**
- ✅ Multiple model IDs accepted
- ✅ Grid size calculation correct
- ✅ Jobs created for all combinations
- ✅ Validation rejects empty model_ids
- ✅ Validation rejects empty tickers
- ✅ Holy Grail criteria passed through

**Test Scenarios:**
1. Single model, single ticker
2. Multiple models, multiple tickers
3. Complex regime configurations
4. Empty model_ids → 400 error
5. Empty tickers → 400 error

### Workflow 4: Full Evolution (Backwards Compatibility)

**Endpoints Tested:**
- `POST /evolve` - Traditional full pipeline

**Validations:**
- ✅ Still creates models
- ✅ Still runs simulations
- ✅ No breaking changes

## Grid Size Calculation Tests

**Formula:** `tickers × thresholds × z_scores × regimes × models`

**Example:**
- 2 tickers
- 4 thresholds
- 5 z-scores
- 7 regime configs
- 3 models

**Calculation:** 2 × 4 × 5 × 7 × 3 = **840 simulations**

**Tested in:** `test_grid_size_calculation()`, `test_run_simulations_for_selected_models()`

## Model Selection Logic Tests

### Toggle Selection
```python
# First click - add
selectedModelIds.add(model_id)

# Second click - remove
selectedModelIds.remove(model_id)
```

### Select All with Filters
```python
for model in models:
    if matches_filters(model):
        selectedModelIds.add(model.id)
```

### Clear All
```python
selectedModelIds.clear()
```

## Filter Logic Tests

### Symbol Filter (partial, case-insensitive)
```python
filter_symbol = "ap"
matches = "ap" in "AAPL".lower()  # True
```

### Algorithm Filter (exact, case-insensitive)
```python
filter_algorithm = "randomforest"
matches = "randomforest" == "randomforest"  # True
matches = "randomforest" == "xgboost"  # False
```

### Accuracy Filter (threshold)
```python
min_accuracy = 60.0
matches = (model.accuracy * 100) >= 60.0
```

## Running All Tests

```bash
# Quick run (unit tests only)
./run_orchestrator_tests.sh

# Full suite (requires services)
docker-compose up -d
python -m pytest tests/unit/test_orchestrator_ui.py tests/integration/test_orchestrator_workflows.py -v

# With coverage report
python -m pytest tests/unit/test_orchestrator_ui.py tests/integration/test_orchestrator_workflows.py -v --cov=orchestrator_service --cov-report=html
```

## Expected Results

**Unit Tests (fast, no services needed):**
- All tests should pass
- Runtime: < 1 second

**Integration Tests (requires services):**
- May have some skips if no models exist yet
- Runtime: 1-3 minutes
- Creates real models in database
- Verifies end-to-end workflows

## Troubleshooting

### "No models in database to test"
- Run train-only workflow manually first
- Or use `sample_trained_models` fixture

### "Cannot browse models"
- Check orchestrator service is running: `docker-compose ps`
- Check database is accessible
- Verify `/models/browse` endpoint responds

### "Simulations not queued"
- Check simulation_service is running
- Verify job queue is accepting jobs
- Check logs: `docker-compose logs simulation_service`

## CI/CD Integration

Add to `.github/workflows/test.yml`:

```yaml
- name: Run Orchestrator Tests
  run: |
    docker-compose up -d
    sleep 10  # Wait for services
    python -m pytest tests/unit/test_orchestrator_ui.py -v
    python -m pytest tests/integration/test_orchestrator_workflows.py -v -m integration
```

## Test Data

Tests use these symbols:
- AAPL (primary test symbol)
- SPY (reference symbol)
- MSFT, QQQ (additional variety)

Tests use these algorithms:
- randomforest (most common)
- xgboost, lightgbm (tree-based)
- ridge, lasso, elasticnet (linear with regularization)
- linearregression (baseline)

## Coverage Goals

- **Unit Tests:** 100% coverage of UI logic functions
- **Integration Tests:** 80%+ coverage of API endpoints
- **End-to-End:** All user workflows tested

## Next Steps

1. Run unit tests to verify logic
2. Start services and run integration tests
3. Test manually in browser
4. Add more edge cases as needed
5. Integrate into CI/CD pipeline

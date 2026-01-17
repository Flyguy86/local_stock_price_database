# Model Registry & Backtest Simulation - Setup Complete

## What's New

### 1. MLflow Model Registry
- **MLflow Server** running at `http://localhost:5000`
- Automatically logs all training runs with:
  - Model artifacts
  - Hyperparameters
  - Training metrics (RMSE, R², MAE, etc.)
  - Feature names and metadata
  - **Feature Permutation Importance** ("polygraph test")

### 2. Model Registry UI
- **New Dashboard** at `http://localhost:8265/registry`
- Features:
  - Browse all registered models
  - View detailed performance metrics
  - Inspect feature permutation importance
  - Transition models between stages (Production/Staging/Archived)
  - Launch backtests with custom parameters

### 3. Backtest Simulation
- **Realistic backtesting** with:
  - Configurable slippage (default: 0.01%)
  - Transaction costs per share (default: $0.001)
  - Position sizing
  - Initial capital
- **Automated execution** via background jobs
- **Results tracking** with job IDs

## Quick Start

### 1. Start Services
```bash
cd /workspaces/local_stock_price_database
docker-compose up --build
```

### 2. Train a Model (Example)
```bash
curl -X POST http://localhost:8265/train/walk-forward \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "GOOGL",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "train_months": 6,
    "test_months": 1,
    "step_months": 1,
    "algorithm": "elasticnet",
    "num_samples": 50
  }'
```

Model will be automatically logged to MLflow!

### 3. Open Model Registry
Visit: `http://localhost:8265/registry`

- **Models Tab**: View all trained models
- **Backtest Tab**: Launch simulations

### 4. View MLflow UI (Optional)
Visit: `http://localhost:5000`
- Experiment tracking
- Model comparison
- Artifact browser

## Model Registry Features

### Browse Models
- Grid view of all registered models
- Shows: name, version, stage, creation date, run ID
- Click any model to view details

### Model Details Modal
When you click a model:
- **Performance Metrics**: RMSE, R², MAE, training time
- **Feature Permutation Importance**: Top 20 features with "polygraph" interpretation
  - 🔴 Critical: High impact (>0.001)
  - 🟡 Moderate: Medium impact (0.0001-0.001)
  - 🟢 Minimal: Low impact (>0)
  - ⚪ No impact: Zero/negative
- **Parameters**: Full hyperparameter configuration
- **Stage Transitions**: Promote to Production, move to Staging, Archive

### Launch Backtest
Parameters you can configure:
- **Model Selection**: Choose any registered model version
- **Ticker**: Test on any symbol (e.g., AAPL, MSFT, TSLA)
- **Date Range**: Start and end dates for backtest
- **Slippage (%)**: Simulated price impact (default: 0.01%)
- **Commission per Share**: Transaction cost (default: $0.001)
- **Initial Capital**: Starting portfolio value (default: $100,000)
- **Position Size (%)**: % of capital per trade (default: 10%)

## API Endpoints

### Model Registry

#### List All Models
```bash
GET /mlflow/models
```
Returns: Array of registered models with versions and stages

#### Get Model Details
```bash
GET /mlflow/model/{model_name}/{version}
```
Returns: Metrics, parameters, tags, permutation importance

#### Transition Model Stage
```bash
POST /mlflow/model/{model_name}/{version}/transition
Content-Type: application/json

{
  "stage": "Production"  # None, Staging, Production, Archived
}
```

### Backtest Simulation

#### Launch Backtest
```bash
POST /backtest/simulate
Content-Type: application/json

{
  "model_name": "walk_forward_elasticnet_GOOGL",
  "model_version": "1",
  "ticker": "AAPL",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "slippage_pct": 0.0001,
  "commission_per_share": 0.001,
  "initial_capital": 100000.0,
  "position_size_pct": 0.1
}
```
Returns: `job_id` for tracking

#### Check Backtest Status
```bash
GET /backtest/status/{job_id}
```
Returns: Status (running/completed/failed), timestamps, error (if any)

#### Get Backtest Results
```bash
GET /backtest/results/{job_id}
```
Returns: Full backtest metrics, predictions, trade log

## Feature Permutation Importance ("Polygraph Test")

### What It Is
- Shuffles each feature and measures performance drop
- Features causing large drops are truly important
- Catches "cheating" features (data leakage)
- Validates feature engineering

### How to Interpret
1. **High importance** (🔴 Critical): Model relies heavily on this feature
   - Example: `close_ratio_QQQ` = 0.00234 → Stock's relative position to market is critical
   
2. **Moderate importance** (🟡): Contributes but not essential
   - Example: `rsi_14_norm` = 0.00045 → RSI helps but model can work without it
   
3. **Minimal/No impact** (🟢⚪): Feature might be redundant
   - Example: `sma_50_norm` = 0.000001 → Barely used by model, consider removing

### Where to Find It
- **Model Registry UI**: Click any model → scroll to "Feature Permutation Test" table
- **MLflow UI**: Artifacts → `permutation_importance.json`
- **API**: `GET /mlflow/model/{name}/{version}` → `permutation_importance` field

## Backtest Workflow

### Critical: Single Source of Truth

**All feature generation uses Ray Data** via `StreamingPreprocessor.calculate_indicators_gpu()`:
- ✅ **Training**: Uses Ray Data pipeline in `create_walk_forward_pipeline()`
- ✅ **Backtesting**: Uses **same Ray Data pipeline** in `backtest_model.py`
- ✅ **Consistency Guaranteed**: Identical calculations, normalization, and versioning

This ensures features in backtesting match training exactly - no discrepancies!

### Step 1: Train Model
Train with walk-forward validation on GOOGL:
```bash
# Via Dashboard UI at http://localhost:8265/
# OR via API (see above)
```

### Step 2: Check Model Registry
Visit `http://localhost:8265/registry`:
- Find your trained model
- Click to view metrics and permutation importance
- Verify it's performing well (R² > 0.3, low RMSE)

### Step 3: Promote to Production (Optional)
- Click model → "Promote to Production" button
- Marks model as production-ready

### Step 4: Launch Backtest
- Switch to "Backtest" tab
- Select model
- Choose ticker (e.g., AAPL to test GOOGL-trained model on Apple)
- Configure slippage, commission, capital
- Click "Launch Backtest"
- Note the `job_id`

### Step 5: Monitor Progress
```bash
curl http://localhost:8265/backtest/status/{job_id}
```

### Step 6: View Results
```bash
curl http://localhost:8265/backtest/results/{job_id}
```

Results include:
- Predictions vs actuals
- Trade-by-trade log
- P&L curve
- Sharpe ratio
- Max drawdown
- Win rate

## Files Modified/Created

### Docker & Infrastructure
- `Dockerfile.mlflow` - MLflow server container
- `docker-compose.yml` - Added MLflow service

### MLflow Integration
- `ray_orchestrator/mlflow_integration.py` - MLflowTracker class with:
  - `log_training_run()` - Log models to MLflow
  - `calculate_permutation_importance()` - Feature polygraph test
  - `get_registered_models()` - List all models
  - `transition_model_stage()` - Promote/demote models

### Training Pipeline
- `ray_orchestrator/trainer.py` - Enhanced with:
  - Auto-logging to MLflow after each training run
  - Permutation importance calculation
  - Model registration

### API Endpoints
- `ray_orchestrator/main.py` - Added:
  - `/registry` - Model Registry UI route
  - `/mlflow/models` - List models API
  - `/mlflow/model/{name}/{version}` - Get model details
  - `/mlflow/model/{name}/{version}/transition` - Change stage
  - `/backtest/simulate` - Launch backtest job
  - `/backtest/status/{job_id}` - Check status
  - `/backtest/results/{job_id}` - Get results

### UI
- `ray_orchestrator/templates/model_registry.html` - Full Model Registry dashboard
  - Models grid with cards
  - Detailed model modal with metrics + permutation importance
  - Backtest launcher form
  - Real-time status updates

### Backtest Script
- `backtest_model.py` - Enhanced with:
  - MLflow model loading (`models:/...` URI support)
  - Realistic slippage and commission simulation
  - Position sizing
  - Results visualization

## Next Steps

1. **Train Your First Model**
   ```bash
   # Use the Training Dashboard at http://localhost:8265/
   ```

2. **Inspect in Model Registry**
   ```bash
   # Visit http://localhost:8265/registry
   ```

3. **Run Backtest on Different Ticker**
   ```bash
   # Use Backtest tab in Model Registry UI
   ```

4. **Analyze Results**
   - Check permutation importance to see which features matter
   - Compare performance across different tickers
   - Use slippage/commission to simulate real trading costs

5. **Iterate**
   - Remove features with minimal permutation importance
   - Retrain with focused feature set
   - Compare new model vs old in MLflow

## Troubleshooting

### MLflow Server Not Starting
```bash
# Check MLflow logs
docker logs mlflow

# Verify port 5000 is free
lsof -i :5000
```

### Models Not Appearing in Registry
```bash
# Check if training completed successfully
docker logs ray_orchestrator

# Verify MLflow can connect
curl http://localhost:5000/health
```

### Backtest Failing
```bash
# Check backtest logs
cat /app/data/ray_checkpoints/backtest_results/{job_id}_status.json

# Verify ticker has data
curl http://localhost:8265/symbols | grep {TICKER}
```

## Architecture

```
┌─────────────────┐      ┌──────────────────┐
│  Training       │─────>│  MLflow Server   │
│  Pipeline       │      │  (Model Registry)│
│  (Ray Tune)     │      └──────────────────┘
└─────────────────┘               │
        │                         │
        v                         v
┌─────────────────┐      ┌──────────────────┐
│  Checkpoints    │      │  Model Registry  │
│  + Metadata     │      │  UI Dashboard    │
└─────────────────┘      └──────────────────┘
        │                         │
        v                         v
┌─────────────────┐      ┌──────────────────┐
│  Backtest       │<─────│  User Selects    │
│  Simulation     │      │  Model + Params  │
│  (Background)   │      └──────────────────┘
└─────────────────┘
        │
        v
┌─────────────────┐
│  Results        │
│  (JSON + CSV)   │
└─────────────────┘
```

## Key Benefits

1. **Centralized Model Management**: All models in one place with versioning
2. **Feature Validation**: Permutation importance catches data leakage and validates engineering
3. **Realistic Backtesting**: Slippage and commission simulate real trading costs
4. **No Manual Tracking**: Auto-logging from training pipeline
5. **Reproducibility**: Every model has full metadata (hyperparams, feature version, fold dates)
6. **Production Workflow**: Clear path from training → staging → production
7. **Cross-Ticker Testing**: Validate model generalization

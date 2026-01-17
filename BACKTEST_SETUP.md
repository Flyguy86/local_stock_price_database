# Backtesting Setup Guide

## Current Issue

You're seeing **"No folds found for symbol AAPL"** because the backtesting system requires walk-forward fold data, which hasn't been generated yet.

### What's Happening

1. **Models exist**: You have trained models in `/app/data/ray_checkpoints/`:
   - `walk_forward_xgboost_AAPL`
   - `walk_forward_xgboost_GOOGL`
   - `walk_forward_elasticnet_GOOGL`

2. **Folds missing**: The walk-forward fold data is missing in `/app/data/walk_forward_folds/`:
   - This directory is currently empty
   - Backtesting requires fold data to run simulations

## Solution: Generate Walk-Forward Folds

### Option 1: Using the UI (Recommended)

1. Navigate to the **Training Dashboard** (port 8265)
2. Go to the **"Date Range"** section
3. Configure your training:
   ```
   Primary Symbol: AAPL (or any symbol you want to backtest)
   Start Date: 2020-01-01 (or your preferred date)
   End Date: 2024-12-31 (or your preferred date)
   Train Months: 12
   Test Months: 3
   Step Months: 3
   ```
4. Click **"🚀 Start Walk-Forward Training"**
   - This will generate folds AND train models
   - Folds will be saved to `/app/data/walk_forward_folds/AAPL/`

### Option 2: Using the API

```bash
curl -X POST http://localhost:8265/streaming/walk_forward \
  -H 'Content-Type: application/json' \
  -d '{
    "primary_symbol": "AAPL",
    "additional_symbols": [],
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "train_months": 12,
    "test_months": 3,
    "step_months": 3,
    "algorithm": "xgboost",
    "n_trials": 10
  }'
```

## Understanding the Workflow

### Complete Training → Backtesting Pipeline

```
1. Generate Folds          POST /streaming/walk_forward
   ↓                       Creates walk-forward folds
   ↓                       Saves to /app/data/walk_forward_folds/{symbol}/
   ↓
2. Train Models            (Automatic with step 1)
   ↓                       Ray Tune hyperparameter search
   ↓                       Saves checkpoints to /app/data/ray_checkpoints/
   ↓
3. Select Best Model       GET /backtest/experiments
   ↓                       Scans checkpoints
   ↓                       Loads model with lowest test_rmse
   ↓
4. Run Backtest            POST /backtest/validate
   ↓                       Loads fold data
   ↓                       Retrains with best hyperparameters
   ↓                       Runs VectorBT simulation
   ↓
5. View Results            Returns Sharpe, drawdown, consistency
                          Displays assessment badges (🟢🟡🟠🔴)
```

## Expected Directory Structure

After running walk-forward training, you should have:

```
/app/data/
├── walk_forward_folds/           # ← Currently EMPTY (this is the problem)
│   ├── AAPL/
│   │   ├── fold_0/
│   │   │   ├── train.parquet
│   │   │   └── test.parquet
│   │   ├── fold_1/
│   │   │   ├── train.parquet
│   │   │   └── test.parquet
│   │   └── fold_2/
│   │       ├── train.parquet
│   │       └── test.parquet
│   └── GOOGL/
│       ├── fold_0/
│       ├── fold_1/
│       └── fold_2/
│
└── ray_checkpoints/              # ✓ Already exists
    ├── walk_forward_xgboost_AAPL/
    │   ├── trial_0/
    │   │   ├── result.json
    │   │   └── params.json
    │   └── trial_1/
    └── walk_forward_xgboost_GOOGL/
```

## What Was Fixed

### 1. Better Error Messages
The backtest endpoint now provides detailed diagnostics:
- Checks if fold base directory exists
- Lists available symbols with fold data
- Provides step-by-step instructions to generate folds

### 2. Model Info Display
Fixed the model dropdown to correctly show:
- Trial ID
- Test RMSE
- Test R²

Previously showed "N/A" due to incorrect field mapping.

### 3. Validation Checks
Added pre-flight checks before backtesting:
- Verify folds exist for selected symbol
- Verify model hyperparameters loaded successfully
- Clear error messages if data is missing

## Quick Start: Generate Folds for AAPL

**Recommended**: Use the UI to avoid JSON formatting errors.

1. Open http://localhost:8265
2. Scroll to "Date Range" section
3. Set:
   - Primary Symbol: `AAPL`
   - Start: `2020-01-01`
   - End: `2024-12-31`
   - Train: `12` months
   - Test: `3` months
   - Step: `3` months
   - Algorithm: `xgboost`
   - Trials: `10`
4. Click "🚀 Start Walk-Forward Training"
5. Wait for completion (check progress in logs)
6. Refresh model dropdown
7. Select `walk_forward_xgboost_AAPL`
8. Click "Run Backtest"

## Troubleshooting

### Model dropdown shows models but details are "N/A"
- **Cause**: `result.json` or `params.json` files missing in checkpoint
- **Fix**: Retrain the model via POST /streaming/walk_forward

### "No folds found for symbol X"
- **Cause**: Walk-forward folds not generated
- **Fix**: Run walk-forward training for that symbol

### Backtest runs but results are poor
- **Expected**: First run establishes baseline
- **Check**: Hyperparameters (n_trials, date range, feature quality)
- **Iterate**: Adjust parameters and retrain

## Next Steps

1. **Generate folds for AAPL** (see Quick Start above)
2. **Verify folds exist**:
   ```bash
   ls -la /app/data/walk_forward_folds/AAPL/
   ```
   Should show `fold_0/`, `fold_1/`, etc.

3. **Refresh model dropdown** in UI
4. **Run backtest** on AAPL
5. **Review results**: Sharpe ratio, max drawdown, consistency scores

## Future Optimizations

Current implementation (quick fix):
- ✅ Works with existing architecture
- ❌ Slower (retrains models per fold)

Planned refactor:
- ✅ Load pre-trained per-fold models from checkpoints
- ✅ 10-100x faster backtesting
- Requires checkpoint structure changes

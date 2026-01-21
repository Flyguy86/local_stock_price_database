# Model Retraining Workflow Guide

## Overview
The retrain workflow allows you to iteratively improve models by excluding low-performing features from previous training runs.

## Features Implemented

### 1. **Improved Experiment Naming**
Experiments now have descriptive names showing key metadata:
- Format: `wf_{algorithm}_{ticker}+{N}ctx_f{features}_tr{train}m_te{test}m`
- Example: `wf_xgboost_GOOGL+2ctx_f156_tr3m_te1m`
  - Algorithm: XGBoost
  - Ticker: GOOGL
  - Context symbols: 2 (e.g., QQQ, VIX)
  - Features: 156 total
  - Training: 3 months
  - Testing: 1 month

### 2. **Model Listing API**
**Endpoint:** `GET /models/list`

Lists all trained models from Ray checkpoints with metadata:
- Algorithm, ticker, feature count
- Test R², RMSE, MAE
- Number of folds
- Context symbols
- Training configuration
- Creation date

**Response:**
```json
{
  "models": [
    {
      "experiment_id": "walk_forward_xgboost_GOOGL",
      "trial_id": "train_on_folds_...",
      "algorithm": "xgboost",
      "ticker": "GOOGL",
      "num_features": 156,
      "avg_test_r2": 0.6234,
      "avg_test_rmse": 0.002134,
      "num_folds": 5,
      "context_symbols": ["QQQ", "VIX"],
      "train_months": 3,
      "test_months": 1,
      "params": {...},
      "config": {...},
      "top_features": [...]
    }
  ]
}
```

### 3. **Feature Importance API**
**Endpoint:** `GET /models/{experiment_id}/{trial_id}/features`

Returns feature importance scores for a specific model:
- Full feature list with importance values
- Features marked as high/low importance (vs median)
- Summary statistics

**Response:**
```json
{
  "feature_importance": [
    {
      "name": "close_log_return",
      "importance": 0.234567,
      "is_low_importance": false
    },
    ...
  ],
  "metadata": {...},
  "summary": {
    "total_features": 156,
    "low_importance_count": 78,
    "median_importance": 0.001234
  }
}
```

### 4. **Training with Feature Exclusion**
**New Parameters:**
- `excluded_features`: List of feature names to exclude from training
- `base_experiment_id`: Reference to original experiment (for tracking)

**Flow:**
1. Features are filtered in `_train_single_fold()` before model training
2. Logs show how many features were excluded
3. Model metadata tracks excluded features for reproducibility

## UI Workflow

### Step 1: Access Retrain Workflow
1. Navigate to http://localhost:8000/ (Training Dashboard)
2. Click **"🔄 Retrain Existing Model"** button (green button next to Start Training)

### Step 2: Select Model
A modal appears showing all trained models in a table:
- **Columns:** Algorithm, Ticker, Features, Folds, Test R², Test RMSE, Context, Date
- **Sorting:** Models sorted by Test R² (best first)
- **Color coding:** R² values color-coded (green=good, yellow=ok, red=poor)

**Actions:**
- Click a row or radio button to select model
- Click **"Next: Configure Features →"**

### Step 3: Review Configuration & Select Features

**Configuration Panel** shows:
- Model summary (algorithm, ticker, metrics)
- All hyperparameters from original training
- Training/test period configuration

**Feature Selection Grid:**
- Shows all features with importance scores
- **Rank:** Feature position (1 = most important)
- **Importance:** Permutation importance value
- **Status:** High (green badge) or Low (red badge)
- **Checkboxes:** ✓ = include in retraining

**Default Behavior:**
- High importance features: **Checked** (included)
- Low importance features: **Unchecked** (excluded)
- Median importance threshold displayed

**Quick Actions:**
- **✓ Select All:** Include all features
- **✗ Clear All:** Exclude all features
- **⭐ High Importance Only:** Only include features above median

**Selection Summary:**
- Shows: X of Y features selected (Z excluded)
- Updates in real-time as you check/uncheck

### Step 4: Start Retraining
1. Review selected features
2. Click **"🚀 Start Retraining with Selected Features"**
3. System submits training job with:
   - Original model configuration
   - Excluded features list
   - Same train/test splits
4. Success message shows Job ID
5. Monitor progress in Ray Dashboard

## Backend Implementation

### Files Modified:

1. **ray_orchestrator/main.py**
   - Added `excluded_features` to `WalkForwardTrainRequest`
   - Added `GET /models/list` endpoint
   - Added `GET /models/{experiment_id}/{trial_id}/features` endpoint
   - Updated training entrypoint to pass excluded_features

2. **ray_orchestrator/trainer.py**
   - Updated experiment naming with metadata
   - Added `excluded_features` parameter to `run_walk_forward_tuning()`
   - Added `excluded_features` parameter to `_train_single_fold()`
   - Features filtered before model training
   - Metadata includes excluded features list

3. **ray_orchestrator/templates/training_dashboard.html**
   - Added retrain modal UI
   - Added JavaScript functions for workflow
   - Added styling for tables and grids

## Example Retrain Workflow

### Scenario: Reduce Overfitting
1. Initial model has 156 features, Test R² = 0.45
2. Review feature importance → 78 features have very low scores
3. Retrain excluding low-importance features
4. New model has 78 features, Test R² = 0.62 ✅

### Benefits:
- **Reduced overfitting:** Fewer noisy features
- **Faster training:** Less data to process
- **Better generalization:** Focus on signal, not noise
- **Lower memory:** Smaller models

## Tips

### XGBoost Incremental Learning vs Fresh Training

**Important:** This retrain workflow creates a **new model from scratch** when you exclude features. Here's why:

#### Two XGBoost Training Modes:

1. **Incremental Learning** (Adding Trees):
   ```python
   # Using xgb_model parameter
   model = xgb.train(params, dtrain, num_boost_round=10, xgb_model='existing_model.json')
   ```
   - **Requires:** Same feature set (same number and order of features)
   - **Effect:** Adds 10 more trees to existing model (e.g., 100 → 110 trees)
   - **Use case:** Extend model capacity without changing features
   - **NOT SUPPORTED** when excluding features (dimension mismatch)

2. **Fresh Training** (What This Workflow Does):
   ```python
   # New model with different features
   model = xgb.train(params, dtrain, num_boost_round=100)
   ```
   - **Allows:** Different feature set (fewer features after exclusion)
   - **Effect:** Trains completely new model from scratch
   - **Use case:** Reduce overfitting by removing low-importance features
   - **CURRENT BEHAVIOR** when features are excluded

#### Process Type 'update' (Refining Trees):
- `process_type='update'` + `update_plugin='refresh'`
- Updates leaf values in existing trees without adding new trees
- **Also requires same feature set**
- Not applicable when excluding features

### When to Retrain:
- Model has low R² despite good hyperparameters
- Many features have near-zero importance
- You suspect overfitting (high train R², low test R²)
- You added new context symbols and want to drop old features

**Note:** For true XGBoost incremental learning (using `xgb_model`), keep all features and extend `num_samples` in regular training instead.

### What to Exclude:
- Features with importance < 0.001 (near zero)
- Features below median importance (automatic in UI)
- Redundant features (e.g., multiple similar indicators)

### What to Keep:
- Top 10-20 features (always keep these)
- Features with consistent importance across folds
- Domain-specific features (e.g., volatility for options)

## Troubleshooting

### "No trained models found"
- Train a model first using regular "Start Walk-Forward Training"
- Check `/app/data/ray_checkpoints/` has checkpoint directories

### "Error loading features"
- Checkpoint may be missing `feature_importance.json`
- Re-run training with latest code (MLflow integration added this file)

### "All features excluded"
- You must select at least 1 feature
- Use "High Importance Only" button to auto-select good features

### Retrain starts but fails immediately
- Check Ray Dashboard logs
- Verify excluded_features list is valid
- Ensure date ranges have data available

## API Testing

### Test model listing:
```bash
curl http://localhost:8000/models/list | jq
```

### Test feature importance:
```bash
# Get experiment_id and trial_id from model list first
curl http://localhost:8000/models/walk_forward_xgboost_GOOGL/train_on_folds_.../features | jq
```

### Test retrain submission:
```bash
curl -X POST http://localhost:8000/train/walk_forward \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["GOOGL"],
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "algorithm": "xgboost",
    "excluded_features": ["feature1", "feature2"]
  }'
```

## Next Steps

1. **Test the UI:** Navigate to dashboard and click "Retrain Existing Model"
2. **Review models:** See if your existing checkpoints appear
3. **Test feature selection:** Try excluding low-importance features
4. **Submit retrain:** Start a new training run with reduced features
5. **Compare results:** Check if R² improves with fewer features

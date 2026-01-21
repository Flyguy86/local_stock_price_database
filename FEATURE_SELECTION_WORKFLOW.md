# Feature Selection and Model Retraining Workflow

## Overview
This document describes the complete feature selection and model retraining workflow that allows iterative model improvement by selecting the most important features from trained models.

## Architecture

### Components
1. **Model Ranking** (`/models/rank`): Ranks models by performance metrics from Ray checkpoints
2. **Trial Details** (`/models/trial-details`): Loads model configuration and feature importance
3. **Training Submission** (`/training/submit`): Submits retrain jobs with selected features
4. **UI Dashboard**: Interactive feature selection and job submission

### Data Flow
```
Ray Checkpoints → Model Ranking → Feature Selection → Job Submission → New Training → New Checkpoints
    ↓                                      ↓                                              ↓
result.json                    feature_importance.json                          Lineage tracking
```

## Workflow Steps

### 1. Rank Existing Models
- **Endpoint**: `POST /models/rank`
- **Input**: `{"experiment_name": "walk_forward_xgboost_GOOGL"}`
- **Data Source**: `/app/data/ray_checkpoints/{experiment}/*/result.json`
- **Output**: Ranked list of trials with metrics:
  - `test_r2`: R² score on test set
  - `test_rmse`: RMSE on test set
  - `train_rmse`: RMSE on train set
  - `overfitting_gap`: |test_rmse - train_rmse| (lower is better)
  - `balanced_score`: test_r2 - (overfitting_gap × 10)

**Key Implementation Details**:
- Reads JSONL format `result.json` files (multiple JSON objects per line)
- Uses **last line** from each result.json for final metrics
- Maps metric names from Ray format to API format
- Sorts by `test_r2` descending

### 2. Select Model for Retraining
- **UI Action**: Click "Select for Retrain" button on ranked model
- **Endpoint**: `GET /models/trial-details?experiment={name}&trial={dir}`
- **Data Sources**:
  - Configuration: `checkpoint_000000/params.json`
  - Metrics: Last line of `result.json`
  - Features: `checkpoint_000000/feature_importance.json`
- **Output**: Complete trial data including 93 features with importance scores

**Feature Importance Format**:
```json
{
  "all_features": [
    {"name": "returns", "importance": 0.153445, "rank": 1},
    {"name": "is_market_open", "importance": 0.141473, "rank": 2},
    ...
  ]
}
```

### 3. Feature Selection
- **UI**: Modal dialog with feature checkboxes
- **Auto-filter**: Features with importance < 0.0001 are unchecked by default
- **Display**: Shows top 10 features by importance
- **Sorting**: Features displayed by importance (highest first)
- **Selection**: User can manually check/uncheck any features

**Default Behavior**:
- All features shown in scrollable list
- Top features pre-selected
- Low-importance features unchecked
- "Select All" / "Clear All" buttons available

### 4. Submit Retrain Job
- **Endpoint**: `POST /training/submit`
- **Input**:
```json
{
  "primary_ticker": "GOOGL",
  "context_tickers": [],
  "algorithm": "xgboost",
  "selected_features": ["returns", "log_returns", ...],
  "fold_config": {
    "train_months": 12,
    "test_months": 3,
    "step_months": 3,
    "start_date": "2023-01-01",
    "end_date": "2024-12-31"
  },
  "hyperparameters": {...},
  "experiment_name": "walk_forward_xgboost_GOOGL_retrain_2026-01-21T08-00-00",
  "parent_experiment": "walk_forward_xgboost_GOOGL",
  "parent_trial": "train_on_folds_0b701_00000_0"
}
```

**Processing**:
1. Load all features from parent checkpoint
2. Calculate excluded features: `all_features - selected_features`
3. Submit Ray job via Job Submission API
4. Pass `excluded_features` to trainer
5. Track lineage via metadata

**Job Metadata**:
```json
{
  "ticker": "GOOGL",
  "algorithm": "xgboost",
  "experiment_name": "walk_forward_xgboost_GOOGL_retrain_...",
  "parent_experiment": "walk_forward_xgboost_GOOGL",
  "parent_trial": "train_on_folds_0b701_00000_0",
  "features_selected": "10",
  "features_excluded": "83",
  "type": "retrain_with_feature_selection"
}
```

### 5. Training Execution
- **Ray Job**: Submitted via JobSubmissionClient
- **Trainer Method**: `run_walk_forward_tuning()`
- **Feature Filtering**: Applied during preprocessing via `excluded_features` parameter
- **Experiment Naming**: Custom name with lineage (`{parent}_retrain_{timestamp}`)
- **Checkpoints**: Saved to `/app/data/ray_checkpoints/{experiment_name}/`

**Training Code**:
```python
results = trainer.run_walk_forward_tuning(
    symbols=["GOOGL"],
    algorithm="xgboost",
    excluded_features=excluded_features,  # Filters features during preprocessing
    experiment_name="walk_forward_xgboost_GOOGL_retrain_2026-01-21T08-00-00",
    ...
)
```

### 6. Monitor Progress
- **Ray Dashboard**: `http://localhost:8265/#/jobs/{job_id}`
- **UI**: Automatically opens Ray Dashboard in new tab after submission
- **Logs**: Real-time job logs available in Ray Dashboard
- **Status**: Track job state (PENDING → RUNNING → SUCCEEDED/FAILED)

## Lineage Tracking

### Experiment Naming Convention
```
{original_experiment}_retrain_{ISO_timestamp}

Example:
walk_forward_xgboost_GOOGL → walk_forward_xgboost_GOOGL_retrain_2026-01-21T08-00-00
```

### Metadata Chain
Each retrained model stores:
- `parent_experiment`: Original experiment name
- `parent_trial`: Original trial directory
- `features_selected`: Number of features used
- `features_excluded`: Number of features excluded

This allows tracking the complete feature selection history.

## Feature Filtering Implementation

### Trainer Support
The `WalkForwardTrainer.run_walk_forward_tuning()` accepts `excluded_features` parameter:

```python
def run_walk_forward_tuning(
    self,
    symbols: List[str],
    excluded_features: Optional[List[str]] = None,
    experiment_name: Optional[str] = None,
    ...
):
```

### Preprocessing Filter
During fold preprocessing, features are filtered:

```python
if excluded_features:
    original_count = len(feature_cols)
    feature_cols = [c for c in feature_cols if c not in excluded_features]
    log.info(f"Excluded {original_count - len(feature_cols)} features ({len(feature_cols)} remaining)")
```

## Performance Metrics

### Ranking Metrics
- **test_r2**: Primary metric for model quality (0-1, higher is better)
- **test_rmse**: Root mean squared error on test set (lower is better)
- **overfitting_gap**: Generalization indicator (lower is better)
- **balanced_score**: Combined metric: `test_r2 - (overfitting_gap × 10)`

### Feature Importance
- **Permutation Importance**: Measured by feature shuffling impact on test RMSE
- **Rank**: Position in importance ranking (1 = most important)
- **Threshold**: Features < 0.0001 importance often add noise

## Testing

### Test Script: `test_submit_retrain.sh`
```bash
#!/bin/bash
# Tests the complete workflow:
# 1. Submits retrain job with 10 selected features
# 2. Verifies job submission response
# 3. Checks job status via Ray API
# 4. Displays Ray Dashboard link
```

### Expected Output
```json
{
  "status": "submitted",
  "job_id": "raysubmit_ABC123...",
  "experiment_name": "walk_forward_xgboost_GOOGL_retrain_2026-01-21T08-00-00",
  "primary_ticker": "GOOGL",
  "features_selected": 10,
  "features_excluded": 83,
  "parent_experiment": "walk_forward_xgboost_GOOGL",
  "parent_trial": "train_on_folds_0b701_00000_0",
  "note": "Job submitted via Ray. View progress at Ray Dashboard (port 8265). Job ID: raysubmit_..."
}
```

## UI Features

### Model Ranking Table
- Displays top 50 models by test_r2
- Shows all key metrics in sortable columns
- "Select for Retrain" button on each row
- Real-time status badges

### Feature Selection Modal
- Scrollable feature list with checkboxes
- Top 10 features displayed prominently
- Importance scores shown for each feature
- Auto-filter toggle for low-importance features
- Select All / Clear All buttons
- Feature count indicator

### Post-Submission
- Success alert with job details
- Automatic Ray Dashboard opening
- Job ID displayed for tracking
- Link to monitor progress

## Error Handling

### Common Issues
1. **Parent checkpoint not found**: Skips exclusion calculation, logs warning
2. **Empty feature selection**: Alert shown, submission blocked
3. **Ray connection failure**: HTTP 500 with error details
4. **Job submission timeout**: Retry or check Ray cluster health

### Logging
All steps logged with context:
```python
log.info(f"Submitting retrain job for {primary_ticker} with {len(selected_features)} selected features")
log.info(f"Parent: {parent_experiment} / {parent_trial}")
log.info(f"Calculated exclusion list: {len(excluded_features)} features to exclude from {len(all_features)} total")
log.info(f"Retrain job submitted: {job_id} for experiment {experiment_name}")
```

## Future Enhancements

### Potential Improvements
1. **Feature Set Versioning**: Track which feature combinations were tested
2. **Auto-tuning**: Automatically find optimal feature subset
3. **A/B Testing**: Compare retrained vs original model performance
4. **Incremental Features**: Add features one at a time to measure impact
5. **Feature Grouping**: Select/deselect related features together
6. **Performance Prediction**: Estimate new model performance before training

### Monitoring Dashboard
- Real-time job progress tracking
- Compare parent vs retrained model metrics
- Feature importance delta visualization
- Training cost analysis (time, compute)

## Files Modified

### Backend
- `ray_orchestrator/main.py`: Added `/training/submit` endpoint
- `ray_orchestrator/trainer.py`: Added `experiment_name` parameter to `run_walk_forward_tuning()`

### Frontend
- `ray_orchestrator/templates/training_dashboard.html`: Enhanced feature selection modal and job submission

### Testing
- `test_submit_retrain.sh`: End-to-end test script

## Summary
This workflow enables data-driven feature selection by:
1. Identifying best-performing models
2. Analyzing feature importance
3. Selecting most impactful features
4. Retraining with reduced feature set
5. Tracking lineage and comparing results

The system maintains full auditability through checkpoint metadata and Ray job tracking.

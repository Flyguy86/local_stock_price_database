# MLflow Visibility Guide

## Overview
This guide explains how feature importance and model rankings are made visible in the MLflow UI, addressing the common issue where this critical training data is hidden in checkpoint artifacts.

## Features

### 1. Feature Importance in MLflow UI

**Problem**: Previously, feature importance was only saved in checkpoint JSON files (`checkpoint_000000/feature_importance.json`), which required manual file downloads to view.

**Solution**: Now feature importance is logged directly to MLflow in multiple formats:

#### What Gets Logged:

1. **MLflow Parameters** (visible in Parameters tab):
   - `feature_names`: Comma-separated list of first 50 features
   - `total_features`: Total number of features used in training

2. **MLflow Metrics** (visible in Metrics tab and plots):
   - `importance_{feature_name}`: Individual metric for each of top 20 features
   - `importance_max`: Highest importance score across all features
   - `importance_mean`: Average importance score
   - `importance_median`: Median importance score
   - `importance_min`: Lowest importance score

3. **MLflow Artifacts** (downloadable files):
   - `permutation_importance.json`: Full feature importance as JSON
   - `feature_importance.csv`: Full feature importance as CSV (easy to open in Excel)

#### How to View in MLflow UI:

1. **Navigate to a Run**:
   ```
   http://localhost:5000
   → Select your experiment
   → Click on any run
   ```

2. **View Parameters**:
   - Click "Parameters" tab
   - See `feature_names` and `total_features`

3. **View Metrics**:
   - Click "Metrics" tab
   - See all `importance_*` metrics
   - Click any metric to see time-series plot (single value for permutation importance)

4. **Compare Runs**:
   - Select multiple runs (checkbox left of run name)
   - Click "Compare" button
   - View side-by-side feature importance metrics

5. **Download Full Data**:
   - Click "Artifacts" tab
   - Download `feature_importance.csv` for full importance table

### 2. Model Ranking & Top Model Tagging

**Problem**: With hundreds of hyperparameter trials, it's hard to identify the best performing models in MLflow UI.

**Solution**: Automatic ranking and tagging of top models.

#### How It Works:

1. **Via API Endpoint**:
   ```bash
   # Rank top 10 models in an experiment by test R²
   curl -X POST "http://localhost:8265/models/rank?experiment_name=wf_xgboost_AAPL_f150_tr3m_te1m&metric=avg_test_r2&top_n=10&ascending=false"
   ```

2. **Via Training Dashboard**:
   - Go to http://localhost:8265
   - Click "🏆 Rank Top Models" button
   - Enter experiment name
   - Select metric to rank by
   - Click "Rank Models Now"

#### Tags Applied:

Each of the top N models gets tagged with:

- `model_rank`: Numeric rank (1, 2, 3, etc.)
- `top_{N}`: Boolean flag (e.g., `top_10: true`)
- `production_candidate`: Only for top 3 models

#### Finding Top Models in MLflow UI:

1. **Filter by Tag**:
   ```
   MLflow UI → Select experiment → Click "Search Runs"
   Enter: tags.top_10 = "true"
   ```

2. **Filter Production Candidates**:
   ```
   Enter: tags.production_candidate = "true"
   ```

3. **Sort by Rank**:
   ```
   Enter: tags.model_rank = "1"  (find #1 ranked model)
   ```

## Usage Examples

### Example 1: View Feature Importance After Training

1. **Start Training**:
   ```bash
   # Submit training job via dashboard or API
   ```

2. **Find Your Run in MLflow**:
   ```
   http://localhost:5000
   → Click experiment "wf_xgboost_AAPL_f150_tr3m_te1m"
   → Click on latest run
   ```

3. **View Top Features**:
   - Click "Metrics" tab
   - Look for `importance_close`, `importance_volume`, etc.
   - These show the permutation importance scores

4. **Download Full Table**:
   - Click "Artifacts" tab
   - Download `feature_importance.csv`
   - Open in Excel/Google Sheets to see all features ranked

### Example 2: Rank Models and Find Best

1. **After Training Completes** (50+ trials):
   ```bash
   # Rank top 10 by test R²
   curl -X POST "http://localhost:8265/models/rank?experiment_name=wf_xgboost_AAPL_f150_tr3m_te1m&metric=avg_test_r2&top_n=10"
   ```

2. **Find Top Model in MLflow**:
   ```
   http://localhost:5000
   → Select experiment
   → Search: tags.model_rank = "1"
   → Click on the run
   → See all metrics and feature importance
   ```

3. **Compare Top 3**:
   - Search: `tags.production_candidate = "true"`
   - Select all 3 runs (checkboxes)
   - Click "Compare"
   - View side-by-side metrics and feature importance

### Example 3: Retrain with Low-Importance Features Removed

1. **Find Model to Retrain**:
   - MLflow UI → Find experiment → Find run ID

2. **Check Feature Importance**:
   - Download `feature_importance.csv`
   - Identify features with importance < 0.001

3. **Retrain via Dashboard**:
   - http://localhost:8265 → Click "🔄 Retrain Existing Model"
   - Enter experiment name and run ID
   - Table shows all features with importance scores
   - Check boxes for low-importance features to exclude
   - Click "Start Retraining"

4. **New Run Logs Updated Features**:
   - Check MLflow for new run
   - Parameter `total_features` will be lower
   - Only selected features logged

## Ranking Metrics

You can rank by any numeric metric logged during training:

| Metric | Direction | Use Case |
|--------|-----------|----------|
| `avg_test_r2` | Higher is better | Overall model fit on test data |
| `avg_test_rmse` | Lower is better | Prediction error magnitude |
| `avg_test_mae` | Lower is better | Absolute error magnitude |
| `avg_train_rmse` | Lower is better | Training fit (watch for overfitting) |
| `num_folds` | - | Check for models that didn't fail mid-training |

## API Reference

### POST /models/rank

Rank models in an experiment and apply tags.

**Parameters**:
- `experiment_name` (str, required): MLflow experiment name
- `metric` (str, default: "avg_test_r2"): Metric to rank by
- `top_n` (int, default: 10): Number of top models to tag
- `ascending` (bool, default: false): Sort direction

**Response**:
```json
{
  "experiment_name": "wf_xgboost_AAPL_f150_tr3m_te1m",
  "metric": "avg_test_r2",
  "top_n": 10,
  "total_ranked": 10,
  "models": [
    {
      "rank": 1,
      "run_id": "abc123...",
      "metric": "avg_test_r2",
      "value": 0.8523,
      "run_name": "trial_42"
    }
  ],
  "summary": {
    "best_value": 0.8523,
    "worst_value": 0.7891,
    "best_run_id": "abc123..."
  }
}
```

## Implementation Details

### Code Locations

1. **Feature Importance Logging**:
   - File: `ray_orchestrator/mlflow_integration.py`
   - Method: `log_permutation_importance()`
   - Logs parameters, metrics, and artifacts

2. **Model Ranking**:
   - File: `ray_orchestrator/mlflow_integration.py`
   - Method: `rank_top_models()`
   - Queries runs, sorts, applies tags

3. **API Endpoint**:
   - File: `ray_orchestrator/main.py`
   - Route: `POST /models/rank`

4. **UI Components**:
   - File: `ray_orchestrator/templates/training_dashboard.html`
   - Button: "🏆 Rank Top Models"
   - Modal with experiment name, metric selection, results table

### Logging Flow

```
Training completes
   ↓
trainer.py calls mlflow_tracker.log_training_run()
   ↓
trainer.py calls mlflow_tracker.calculate_permutation_importance()
   ↓
trainer.py calls mlflow_tracker.log_permutation_importance()
   ↓
log_permutation_importance() logs:
   - Parameters: feature_names, total_features
   - Metrics: importance_* for top 20 features, summary stats
   - Artifacts: JSON and CSV files
   ↓
MLflow UI now shows all this data
```

### Ranking Flow

```
User clicks "🏆 Rank Top Models" in dashboard
   ↓
JavaScript calls POST /models/rank
   ↓
API calls mlflow_tracker.rank_top_models()
   ↓
rank_top_models() queries MLflow API:
   - Get all runs in experiment
   - Sort by specified metric
   - Take top N runs
   - Apply tags: model_rank, top_N, production_candidate
   ↓
Returns ranked list to UI
   ↓
UI displays table with ranks and tags
   ↓
MLflow UI now shows tags on runs (filterable/searchable)
```

## Troubleshooting

### Feature Importance Not Showing

**Symptom**: Parameters/metrics tabs empty or missing importance data.

**Check**:
1. Verify training completed successfully (no errors in Ray logs)
2. Check MLflow run logs for warnings about logging failures
3. Verify permutation importance was calculated (check Ray logs for "Permutation importance logged")
4. Check if sample size was sufficient (needs >= 10 samples)

**Fix**:
```python
# In trainer.py, verify this code executed:
if len(X_sample) >= 10:
    importance_df = mlflow_tracker.calculate_permutation_importance(...)
    mlflow_tracker.log_permutation_importance(run_id, importance_df)
```

### Model Ranking Failed

**Symptom**: "No models found" or 404 error.

**Check**:
1. Experiment name is correct (exact match, case-sensitive)
2. Experiment has completed runs with the specified metric
3. MLflow server is accessible (http://localhost:5000)

**Fix**:
```bash
# Check experiment exists
curl http://localhost:5000/api/2.0/mlflow/experiments/search

# Check runs in experiment
curl http://localhost:5000/api/2.0/mlflow/runs/search -d '{"experiment_ids":["1"]}'
```

### Tags Not Appearing in MLflow UI

**Symptom**: Ranking succeeds but tags don't show in UI.

**Check**:
1. Refresh MLflow UI page
2. Check run details → Tags section
3. Verify tags via API:
   ```bash
   curl http://localhost:5000/api/2.0/mlflow/runs/get?run_id=<run_id>
   ```

**Fix**: Tags are applied asynchronously. Wait 5-10 seconds and refresh.

## Best Practices

1. **Run Ranking After Training Completes**:
   - Wait for all hyperparameter trials to finish
   - Rank once to tag all top models
   - Re-rank if you add more trials later

2. **Use Consistent Metrics**:
   - For regression: `avg_test_r2` (higher is better)
   - Avoid ranking by training metrics (can indicate overfitting)

3. **Download Feature Importance CSV**:
   - Easier to analyze than JSON
   - Can sort/filter in spreadsheet
   - Share with team members

4. **Tag Production Models**:
   - Top 3 automatically tagged as production candidates
   - Add custom tags via MLflow UI for deployment tracking

5. **Clean Up Old Experiments**:
   - Use MLflow UI to delete failed/test experiments
   - Keeps model registry organized
   - Improves ranking performance

## Next Steps

1. **Automated Ranking**: Add scheduled job to rank models after training
2. **Slack Notifications**: Send top model metrics to team channel
3. **Feature Importance Plots**: Generate bar charts in MLflow artifacts
4. **SHAP Values**: Add SHAP importance alongside permutation importance
5. **Model Registry Integration**: Auto-promote top models to registry stages

## References

- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Permutation Importance (sklearn)](https://scikit-learn.org/stable/modules/permutation_importance.html)

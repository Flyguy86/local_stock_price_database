# MLflow Visibility Features - Quick Summary

## What Changed

### ✅ Feature Importance Now Visible in MLflow UI

**Before**: Feature importance only in checkpoint JSON files (hidden)
**After**: Feature importance in MLflow Parameters, Metrics, and Artifacts tabs

**What's Logged**:
- **Parameters**: `feature_names` (first 50), `total_features` (count)
- **Metrics**: `importance_{feature}` for top 20 features, plus summary stats
- **Artifacts**: Full CSV and JSON files for download

**Where**: `ray_orchestrator/mlflow_integration.py` → `log_permutation_importance()`

---

### 🏆 Model Ranking & Tagging

**New Feature**: Identify and tag top N models in an experiment

**Tags Applied**:
- `model_rank`: "1", "2", "3", etc.
- `top_10`: "true" (or top_N based on request)
- `production_candidate`: "true" (top 3 only)

**Usage**:
```bash
# Via API
curl -X POST "http://localhost:8265/models/rank?experiment_name=wf_xgboost_AAPL_f150_tr3m_te1m&metric=avg_test_r2&top_n=10"

# Via UI
http://localhost:8265 → Click "🏆 Rank Top Models"
```

**Where**: 
- Backend: `ray_orchestrator/mlflow_integration.py` → `rank_top_models()`
- API: `ray_orchestrator/main.py` → `POST /models/rank`
- UI: `training_dashboard.html` → "🏆 Rank Top Models" button + modal

---

## Files Modified

1. **ray_orchestrator/mlflow_integration.py**:
   - Enhanced `log_permutation_importance()` to log params/metrics/artifacts
   - Added `rank_top_models()` method

2. **ray_orchestrator/main.py**:
   - Added `POST /models/rank` endpoint

3. **ray_orchestrator/templates/training_dashboard.html**:
   - Added "🏆 Rank Top Models" button
   - Added ranking modal with experiment selector and results table
   - Added JavaScript functions: `showRankModelsDialog()`, `closeRankModelsDialog()`, `rankModels()`

4. **Documentation**:
   - Created `MLFLOW_VISIBILITY_GUIDE.md` (comprehensive guide)
   - Created `MLFLOW_FEATURES_SUMMARY.md` (this file)

---

## How to Use

### View Feature Importance

1. Train a model (via dashboard or API)
2. Go to MLflow UI: http://localhost:5000
3. Select experiment → Select run
4. **Parameters tab**: See feature names and count
5. **Metrics tab**: See top 20 feature importance scores
6. **Artifacts tab**: Download `feature_importance.csv` for full list

### Rank Models

1. After training completes, go to: http://localhost:8265
2. Click **"🏆 Rank Top Models"** button
3. Enter experiment name (e.g., `wf_xgboost_AAPL_f150_tr3m_te1m`)
4. Select metric (default: `avg_test_r2`)
5. Set top N (default: 10)
6. Click **"Rank Models Now"**
7. View results table with ranks and tags
8. Go to MLflow UI → Search by tag: `tags.top_10 = "true"`

### Compare Top Models

1. In MLflow UI, search: `tags.production_candidate = "true"`
2. Select top 3 models (checkboxes)
3. Click **"Compare"** button
4. View side-by-side metrics and feature importance

---

## Testing Checklist

- [ ] Start training job with at least 10 samples
- [ ] Check MLflow run has `feature_names` parameter
- [ ] Check MLflow run has `importance_*` metrics
- [ ] Download `feature_importance.csv` artifact
- [ ] Run model ranking via API or UI
- [ ] Search MLflow for `tags.top_10 = "true"`
- [ ] Verify top 3 have `tags.production_candidate = "true"`
- [ ] Compare multiple runs side-by-side in MLflow

---

## Example Workflow

```bash
# 1. Start training
curl -X POST http://localhost:8265/train/walk_forward \
  -H "Content-Type: application/json" \
  -d '{"preprocessing_config": {...}, "num_samples": 50}'

# 2. Wait for completion (check Ray Dashboard)

# 3. Rank models
curl -X POST "http://localhost:8265/models/rank?experiment_name=wf_xgboost_AAPL_f150_tr3m_te1m&metric=avg_test_r2&top_n=10"

# 4. View in MLflow
# Go to http://localhost:5000
# Search: tags.model_rank = "1"
# Click on run → See all feature importance data
```

---

## Benefits

✅ **No more digging through checkpoint files**  
✅ **Feature importance visible at a glance in MLflow UI**  
✅ **Easy comparison of features across runs**  
✅ **Downloadable CSV for offline analysis**  
✅ **Automatic identification of best models**  
✅ **Searchable/filterable tags for production deployment**  
✅ **Side-by-side comparison of top performers**

---

## Next Steps (Future Enhancements)

1. Add feature importance plots (bar charts) as artifacts
2. Add SHAP values alongside permutation importance
3. Auto-rank models after training completes
4. Send Slack notifications with top model metrics
5. Auto-promote top model to MLflow Model Registry staging
6. Add "Deploy to Production" button for top-ranked models

---

## References

- Full guide: [MLFLOW_VISIBILITY_GUIDE.md](./MLFLOW_VISIBILITY_GUIDE.md)
- MLflow docs: https://mlflow.org/docs/latest/
- Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html

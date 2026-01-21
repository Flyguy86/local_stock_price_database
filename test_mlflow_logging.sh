#!/bin/bash
# Quick test to verify MLflow feature importance logging

echo "🧪 Testing MLflow Feature Importance Logging"
echo "=============================================="
echo ""
echo "Starting a minimal training run to test new MLflow logging..."
echo ""

# Submit a minimal training job (1 trial, short period)
curl -X POST http://localhost:8265/train/walk_forward \
  -H "Content-Type: application/json" \
  -d '{
    "preprocessing_config": {
      "symbols": ["GOOGL"],
      "start_date": "2024-01-01",
      "end_date": "2024-03-31",
      "train_months": 1,
      "test_months": 1,
      "step_months": 1,
      "windows": [20],
      "resampling_timeframes": ["1min"],
      "context_symbols": []
    },
    "hyperparameter_config": {
      "algorithm": "xgboost",
      "learning_rate": [0.05],
      "max_depth": [3],
      "n_estimators": [50]
    },
    "num_samples": 1
  }' | jq .

echo ""
echo "✅ Training job submitted!"
echo ""
echo "Next steps:"
echo "1. Wait ~5 minutes for training to complete"
echo "2. Check Ray Dashboard: http://localhost:8265"
echo "3. Once complete, go to MLflow: http://localhost:5000"
echo "4. Look for the newest run with today's date"
echo "5. Click on it and check:"
echo "   - Parameters tab → Should see 'feature_names' and 'total_features'"
echo "   - Metrics tab → Should see 'importance_*' for top 20 features"
echo "   - Artifacts tab → Should see 'feature_importance.csv'"
echo ""

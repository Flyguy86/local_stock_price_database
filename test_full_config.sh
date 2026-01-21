#!/bin/bash
echo "🔄 Restarting ray_orchestrator with full config support..."
docker-compose restart ray_orchestrator

echo "⏳ Waiting for service..."
for i in {1..30}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "✅ Service ready after $i seconds"
        break
    fi
    sleep 1
done

echo ""
echo "Testing with ALL parameters from parent trial..."
curl -s -X POST "http://localhost:8100/training/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "primary_ticker": "GOOGL",
    "context_tickers": [],
    "algorithm": "xgboost",
    "selected_features": ["returns", "log_returns", "is_market_open"],
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "train_months": 12,
    "test_months": 3,
    "step_months": 3,
    "windows": [5, 10, 20, 60],
    "resampling_timeframes": ["1min", "5min", "15min", "30min", "1h", "4h", "1d"],
    "hyperparameters": {
      "colsample_bytree": 0.8582,
      "learning_rate": 0.0226,
      "max_depth": 6,
      "n_estimators": 300,
      "subsample": 0.9261
    },
    "num_samples": 2,
    "experiment_name": "walk_forward_xgboost_GOOGL_retrain_FULL_CONFIG_TEST",
    "parent_experiment": "walk_forward_xgboost_GOOGL",
    "parent_trial": "train_on_folds_0b701_00044_44_colsample_bytree=0.8582,learning_rate=0.0226,max_depth=6,n_estimators=300,subsample=0.9261_2026-01-21_01-55-39"
  }' | python3 -m json.tool

echo ""
echo "✅ Check above - should see job_id and proper ticker (GOOGL, not None)"

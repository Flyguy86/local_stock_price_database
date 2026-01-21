#!/bin/bash
echo "🔄 Restarting with MLflow disabled for retrain jobs..."
docker-compose restart ray_orchestrator

echo "⏳ Waiting..."
for i in {1..30}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "✅ Ready after $i seconds"
        break
    fi
    sleep 1
done

echo ""
echo "Testing retrain job with MLflow disabled..."
RESPONSE=$(curl -s -X POST "http://localhost:8100/training/submit" \
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
    "hyperparameters": {},
    "num_samples": 2,
    "experiment_name": "walk_forward_xgboost_GOOGL_retrain_NO_MLFLOW",
    "parent_experiment": "walk_forward_xgboost_GOOGL",
    "parent_trial": "train_on_folds_0b701_00044_44_colsample_bytree=0.8582,learning_rate=0.0226,max_depth=6,n_estimators=300,subsample=0.9261_2026-01-21_01-55-39"
  }')

echo "$RESPONSE" | python3 -m json.tool
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null)

if [ ! -z "$JOB_ID" ]; then
    echo ""
    echo "✅ Job submitted: $JOB_ID"
    echo "Monitoring for 30 seconds..."
    
    for i in {1..10}; do
        sleep 3
        STATUS=$(curl -s "http://localhost:8100/train/job/$JOB_ID" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('status')); logs=d.get('logs',''); print('GOOGL found!' if 'GOOGL' in logs else ''); print('Training started!' if 'Starting walk-forward' in logs else '')" 2>/dev/null)
        echo "  Check $i: $STATUS"
    done
fi

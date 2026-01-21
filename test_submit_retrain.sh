#!/bin/bash
set -e

echo "🔄 Restarting ray_orchestrator service..."
docker-compose restart ray_orchestrator

echo "⏳ Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "✅ Service ready after $i seconds"
        break
    fi
    sleep 1
done

echo ""
echo "📋 Testing /training/submit endpoint..."
echo "Submitting retrain job with 10 selected features..."

RESPONSE=$(curl -s -X POST "http://localhost:8100/training/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "primary_ticker": "GOOGL",
    "context_tickers": [],
    "algorithm": "xgboost",
    "selected_features": [
      "returns",
      "log_returns",
      "is_market_open",
      "vix_close_lag1",
      "volatility",
      "rsi_14",
      "volume_ratio",
      "price_momentum_5",
      "bid_ask_spread",
      "trade_intensity"
    ],
    "fold_config": {
      "train_months": 12,
      "test_months": 3,
      "step_months": 3,
      "start_date": "2023-01-01",
      "end_date": "2024-12-31"
    },
    "num_samples": 2,
    "experiment_name": "walk_forward_xgboost_GOOGL_retrain_2026-01-21T08-00-00",
    "parent_experiment": "walk_forward_xgboost_GOOGL",
    "parent_trial": "train_on_folds_0b701_00000_0"
  }')

echo ""
echo "📤 API Response:"
echo "$RESPONSE" | python3 -m json.tool

echo ""
echo "🔍 Extracting job_id..."
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', 'NOT_FOUND'))")

if [ "$JOB_ID" != "NOT_FOUND" ] && [ ! -z "$JOB_ID" ]; then
    echo "✅ Job submitted successfully!"
    echo "   Job ID: $JOB_ID"
    echo "   View in Ray Dashboard: http://localhost:8265/#/jobs/$JOB_ID"
    echo ""
    echo "⏳ Checking job status..."
    sleep 2
    curl -s "http://localhost:8100/train/job/$JOB_ID" | python3 -m json.tool | head -20
else
    echo "❌ Job submission failed - no job_id in response"
    exit 1
fi

echo ""
echo "✅ Test complete!"

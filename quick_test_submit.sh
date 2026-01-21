#!/bin/bash
set -e

echo "🧪 Testing /training/submit endpoint..."
echo ""

# Submit a small retrain job
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
      "time_sin",
      "dist_sma_50"
    ],
    "fold_config": {
      "train_months": 12,
      "test_months": 3,
      "step_months": 3,
      "start_date": "2023-01-01",
      "end_date": "2024-12-31"
    },
    "num_samples": 2,
    "experiment_name": "walk_forward_xgboost_GOOGL_retrain_TEST",
    "parent_experiment": "walk_forward_xgboost_GOOGL",
    "parent_trial": "train_on_folds_0b701_00000_0_colsample_bytree=0.9980,learning_rate=0.0279,max_depth=6,n_estimators=259,subsample=0.9926_2026-01-21_01-53-08"
  }')

echo "📤 API Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Check if we got a job_id
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))" 2>/dev/null || echo "")

if [ ! -z "$JOB_ID" ]; then
    echo "✅ SUCCESS! Job submitted with ID: $JOB_ID"
    echo "   View at: http://localhost:8265/#/jobs/$JOB_ID"
    echo ""
    echo "⏳ Checking job status..."
    sleep 2
    curl -s "http://localhost:8100/train/job/$JOB_ID" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Status: {data.get('status')}\"); print(f\"Logs preview: {data.get('logs', '')[:200]}...\")" || echo "Job status endpoint unavailable"
else
    echo "❌ FAILED - No job_id in response"
    exit 1
fi

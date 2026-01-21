#!/bin/bash
echo "Testing training submission endpoint..."
docker-compose restart ray_orchestrator
echo "Waiting for service..."
for i in {1..20}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "Service ready after $i seconds"
        break
    fi
    sleep 1
done

echo ""
echo "Testing /training/submit endpoint..."
curl -s -X POST "http://localhost:8100/training/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "primary_ticker": "GOOGL",
    "context_tickers": [],
    "algorithm": "xgboost",
    "selected_features": ["returns", "log_returns", "is_market_open"],
    "experiment_name": "walk_forward_xgboost_GOOGL_retrain_2026-01-21T08-00-00",
    "parent_experiment": "walk_forward_xgboost_GOOGL",
    "parent_trial": "train_on_folds_0b701_00000_0"
  }' | python3 -m json.tool

echo ""
echo "✅ Endpoint test complete!"

#!/bin/bash
echo "Testing API endpoint directly..."
sleep 2

# Test with proper URL encoding
TRIAL_DIR="train_on_folds_0b701_00000_0_colsample_bytree=0.9977,learning_rate=0.0494,max_depth=9,n_estimators=300,subsample=0.8174_2026-01-21_01-18-09"
ENCODED_TRIAL_DIR=$(printf %s "$TRIAL_DIR" | jq -sRr @uri)

echo "Encoded trial_dir: $ENCODED_TRIAL_DIR"
echo ""

curl -s "http://localhost:8100/models/trial-details?experiment_name=walk_forward_xgboost_GOOGL&trial_dir=$ENCODED_TRIAL_DIR" 2>&1

echo ""
echo ""
echo "Checking orchestrator logs for errors..."
docker logs ray_orchestrator --tail 50 | grep -A 5 "trial-details"

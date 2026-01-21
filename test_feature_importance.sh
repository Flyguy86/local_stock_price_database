#!/bin/bash
echo "Testing feature importance extraction from checkpoint..."
echo ""

docker-compose restart ray_orchestrator
echo "Waiting for service to be ready..."

# Poll health endpoint until ready (max 30 seconds)
for i in {1..30}; do
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "Service ready after $i seconds"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Service did not start within 30 seconds"
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo ""

TRIAL_DIR="train_on_folds_0b701_00000_0_colsample_bytree=0.9977,learning_rate=0.0494,max_depth=9,n_estimators=300,subsample=0.8174_2026-01-21_01-18-09"
ENCODED=$(printf %s "$TRIAL_DIR" | jq -sRr @uri)

echo "Fetching trial details..."
RESPONSE=$(curl -s "http://localhost:8100/models/trial-details?experiment_name=walk_forward_xgboost_GOOGL&trial_dir=$ENCODED")

if [ -z "$RESPONSE" ]; then
    echo "❌ Empty response from API. Checking logs..."
    docker logs ray_orchestrator --tail 30
    exit 1
fi

echo "$RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'detail' in data:
        print(f'❌ Error: {data[\"detail\"]}')
        sys.exit(1)
    print(f'Trial: {data[\"trial_dir\"][:50]}...')
    print(f'Config keys: {list(data[\"config\"].keys())[:5]}...')
    print(f'Metrics: test_r2={data[\"metrics\"].get(\"test_r2\", 0):.4f}')
    print(f'Feature importance count: {len(data[\"feature_importance\"])}')
    print()
    print('Top 10 features by importance:')
    for feat in data['feature_importance'][:10]:
        print(f'  {feat[\"rank\"]:2d}. {feat[\"feature\"]:30s} {feat[\"importance\"]:.6f}')
except json.JSONDecodeError as e:
    print(f'❌ JSON decode error: {e}')
    print('Response was:', sys.stdin.read()[:200])
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Feature importance loaded successfully from checkpoint!"
fi

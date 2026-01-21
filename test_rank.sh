#!/bin/bash
set -e
cd /workspaces/local_stock_price_database
echo "🔄 Restarting orchestrator..."
docker-compose restart ray_orchestrator
echo "⏳ Waiting for service to be ready..."
sleep 5
echo ""
echo "📊 Testing model ranking by test_r2 (specific per-model R²)..."
docker exec ray_orchestrator python3 -c "
from ray_orchestrator.mlflow_integration import MLflowTracker
tracker = MLflowTracker()
results = tracker.rank_top_models('walk_forward_xgboost_GOOGL', 'test_r2', 5, False)
print(f'✅ Found {len(results)} models\n')
if results:
    print('Top 5 models ranked by test R²:')
    for r in results:
        r2 = r.get('test_r2', r.get('value', 0))
        rmse_test = r.get('test_rmse', 0)
        rmse_train = r.get('train_rmse', 0)
        print(f'  #{r[\"rank\"]}: {r[\"trial_id\"]}')
        print(f'          R²={r2:.4f}, test_RMSE={rmse_test:.6f}, train_RMSE={rmse_train:.6f}')
else:
    print('❌ No results returned')
    exit(1)
"
echo ""
echo "✅ Test complete!"


#!/bin/bash
cd /workspaces/local_stock_price_database
docker-compose restart ray_orchestrator
echo "Waiting for orchestrator to restart..."
sleep 5
docker exec ray_orchestrator python3 -c "
from ray_orchestrator.mlflow_integration import MLflowTracker
tracker = MLflowTracker()
results = tracker.rank_top_models('walk_forward_xgboost_GOOGL', 'avg_test_r2', 5, False)
print(f'Found {len(results)} results')
if results:
    print('Top 3 models:')
    for i, r in enumerate(results[:3], 1):
        print(f'{i}. trial={r[\"trial_id\"]}, R²={r[\"avg_test_r2\"]:.4f}, test_RMSE={r[\"avg_test_rmse\"]:.4f}, train_RMSE={r[\"avg_train_rmse\"]:.4f}')
else:
    print('No results')
"

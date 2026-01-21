#!/bin/bash
echo "Restarting orchestrator..."
docker-compose restart ray_orchestrator
sleep 5
echo ""
echo "Testing overfitting gap calculation..."
docker exec ray_orchestrator python3 -c "
from ray_orchestrator.mlflow_integration import MLflowTracker
tracker = MLflowTracker()
results = tracker.rank_top_models('walk_forward_xgboost_GOOGL', 'test_r2', 3, False)
print(f'Found {len(results)} models\n')
for r in results:
    print(f'Rank {r[\"rank\"]}: {r[\"trial_id\"]}')
    print(f'  test_r2:         {r.get(\"test_r2\", 0):.6f}')
    print(f'  test_rmse:       {r.get(\"test_rmse\", 0):.6f}')
    print(f'  train_rmse:      {r.get(\"train_rmse\", 0):.6f}')
    print(f'  overfitting_gap: {r.get(\"overfitting_gap\", 0):.6f}')
    print()
"

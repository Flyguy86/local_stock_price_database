#!/bin/bash
set -x
echo "Starting validation test..."
docker exec ray_orchestrator python3 -c '
from ray_orchestrator.mlflow_integration import MLflowTracker
print("Creating tracker...")
tracker = MLflowTracker()
print("Calling rank_top_models...")
results = tracker.rank_top_models("walk_forward_xgboost_GOOGL", "test_r2", 3, False)
print(f"\n=== RESULTS ===")
print(f"Total: {len(results)}")
for r in results:
    r2 = r.get("test_r2", r.get("value"))
    test_rmse = r.get("test_rmse", 0)
    train_rmse = r.get("train_rmse", 0)
    print(f"Rank {r[\"rank\"]}: trial={r[\"trial_id\"]}")
    print(f"  R2={r2:.4f}, test_RMSE={test_rmse:.6f}, train_RMSE={train_rmse:.6f}")
'
echo "Test complete"

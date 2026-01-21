docker-compose restart ray_orchestrator && sleep 5 && docker exec ray_orchestrator python3 << 'PYEOF'
from ray_orchestrator.mlflow_integration import MLflowTracker
tracker = MLflowTracker()
results = tracker.rank_top_models("walk_forward_xgboost_GOOGL", "test_r2", 5, False)
print(f"\n{'='*60}")
print(f"VALIDATION TEST RESULTS")
print(f"{'='*60}")
print(f"Total models ranked: {len(results)}")
if results:
    print(f"\nTop 5 models by test_r2:")
    for r in results:
        print(f"  Rank #{r['rank']}: {r['trial_id']}")
        print(f"    test_r2:    {r.get('test_r2', 'MISSING'):.4f}" if isinstance(r.get('test_r2'), (int, float)) else f"    test_r2:    {r.get('test_r2', 'MISSING')}")
        print(f"    test_rmse:  {r.get('test_rmse', 'MISSING'):.6f}" if isinstance(r.get('test_rmse'), (int, float)) else f"    test_rmse:  {r.get('test_rmse', 'MISSING')}")
        print(f"    train_rmse: {r.get('train_rmse', 'MISSING'):.6f}" if isinstance(r.get('train_rmse'), (int, float)) else f"    train_rmse: {r.get('train_rmse', 'MISSING')}")
    print(f"\n{'='*60}")
    print(f"✅ TEST PASSED - Found {len(results)} models with metrics")
    print(f"{'='*60}")
else:
    print("\n❌ TEST FAILED - No results returned")
    import sys
    sys.exit(1)
PYEOF

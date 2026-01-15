#!/usr/bin/env python3
"""
Verify that ElasticNet grid search is working correctly.
This script tests the full training pipeline including:
1. Submitting a grid search training job
2. Polling for completion
3. Verifying parent model and child models are created correctly
4. Checking that parent_model_id is set on children

Run inside the orchestrator or training container:
  docker exec -it local_stock_price_database-orchestrator-1 python /app/verify_grid_search.py
"""
import asyncio
import os
import sys
import json
import time
import httpx

# Configuration
TRAINING_SERVICE_URL = os.environ.get("TRAINING_SERVICE_URL", "http://training_service:8200")
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory"
)

# Test parameters
TEST_SYMBOL = "AAPL"
TEST_ALGORITHM = "elasticnet_regression"
TEST_TARGET_COL = "close"
TEST_TIMEFRAME = "1m"
ALPHA_GRID = [0.001, 0.01, 0.1]  # Small grid for faster testing
L1_RATIO_GRID = [0.5, 0.9]       # 3 x 2 = 6 combinations


async def submit_training_job(save_all_grid_models: bool = True):
    """Submit a grid search training job to the training service."""
    print("\n" + "=" * 60)
    print("Step 1: Submitting Grid Search Training Job")
    print("=" * 60)
    
    payload = {
        "symbol": TEST_SYMBOL,
        "algorithm": TEST_ALGORITHM,
        "target_col": TEST_TARGET_COL,
        "params": {
            "fit_intercept": True,
            "max_iter": 1000,
        },
        "timeframe": TEST_TIMEFRAME,
        "alpha_grid": ALPHA_GRID,
        "l1_ratio_grid": L1_RATIO_GRID,
        "save_all_grid_models": save_all_grid_models,
        "cv_folds": 3,  # Fewer folds for faster testing
    }
    
    print(f"  Symbol: {TEST_SYMBOL}")
    print(f"  Algorithm: {TEST_ALGORITHM}")
    print(f"  Alpha Grid: {ALPHA_GRID}")
    print(f"  L1 Ratio Grid: {L1_RATIO_GRID}")
    print(f"  Expected combinations: {len(ALPHA_GRID) * len(L1_RATIO_GRID)}")
    print(f"  Save all grid models: {save_all_grid_models}")
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{TRAINING_SERVICE_URL}/train",
            json=payload
        )
        
        if response.status_code != 200:
            print(f"  ✗ Failed to submit job: {response.status_code}")
            print(f"    Response: {response.text}")
            return None
        
        result = response.json()
        model_id = result.get("model_id") or result.get("id")
        print(f"  ✓ Job submitted successfully")
        print(f"    Model ID: {model_id}")
        return model_id


async def poll_for_completion(model_id: str, timeout: int = 120):
    """Poll the training service until the job completes or times out."""
    print("\n" + "=" * 60)
    print("Step 2: Waiting for Training to Complete")
    print("=" * 60)
    
    start_time = time.time()
    last_status = None
    
    async with httpx.AsyncClient(timeout=10) as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get(f"{TRAINING_SERVICE_URL}/models/{model_id}")
                
                if response.status_code == 200:
                    model = response.json()
                    status = model.get("status")
                    
                    if status != last_status:
                        elapsed = int(time.time() - start_time)
                        print(f"  [{elapsed}s] Status: {status}")
                        last_status = status
                    
                    if status == "completed":
                        print(f"  ✓ Training completed successfully!")
                        return model
                    elif status == "failed":
                        error = model.get("error_message", "Unknown error")
                        print(f"  ✗ Training failed: {error}")
                        return model
                
            except Exception as e:
                print(f"  Warning: Poll error: {e}")
            
            await asyncio.sleep(2)
    
    print(f"  ✗ Timeout after {timeout}s")
    return None


async def verify_model(model_id: str):
    """Verify the completed model has correct data."""
    print("\n" + "=" * 60)
    print("Step 3: Verifying Parent Model")
    print("=" * 60)
    
    import asyncpg
    
    conn = await asyncpg.connect(POSTGRES_URL)
    try:
        # Get the parent model
        parent = await conn.fetchrow(
            "SELECT * FROM models WHERE id = $1",
            model_id
        )
        
        if not parent:
            print(f"  ✗ Parent model {model_id} not found!")
            return False
        
        print(f"  Model ID: {parent['id']}")
        print(f"  Status: {parent['status']}")
        print(f"  Algorithm: {parent['algorithm']}")
        print(f"  Symbol: {parent['symbol']}")
        
        # Check metrics
        metrics = parent['metrics']
        if metrics:
            if isinstance(metrics, str):
                metrics = json.loads(metrics)
            print(f"  Metrics: {json.dumps(metrics, indent=4)[:200]}...")
            
            if 'r2' in metrics or 'R2' in metrics or 'test_r2' in metrics:
                print(f"  ✓ Metrics contain R² score")
            else:
                print(f"  ⚠ Metrics might be missing R² score")
        else:
            print(f"  ⚠ No metrics found")
        
        # Check artifact path
        if parent['artifact_path']:
            print(f"  Artifact: {parent['artifact_path']}")
            print(f"  ✓ Model artifact saved")
        else:
            print(f"  ⚠ No artifact path")
        
        # Check feature columns
        feature_cols = parent['feature_cols']
        if feature_cols:
            if isinstance(feature_cols, str):
                feature_cols = json.loads(feature_cols)
            print(f"  Feature columns: {len(feature_cols)} features")
            print(f"  ✓ Feature columns saved")
        else:
            print(f"  ⚠ No feature columns")
        
        return parent['status'] == 'completed'
        
    finally:
        await conn.close()


async def verify_child_models(parent_model_id: str):
    """Verify child grid models were created with parent_model_id set."""
    print("\n" + "=" * 60)
    print("Step 4: Verifying Child Grid Models")
    print("=" * 60)
    
    import asyncpg
    
    conn = await asyncpg.connect(POSTGRES_URL)
    try:
        # Get child models
        children = await conn.fetch(
            "SELECT id, status, parent_model_id, hyperparameters, metrics, is_grid_member "
            "FROM models WHERE parent_model_id = $1 ORDER BY created_at",
            parent_model_id
        )
        
        expected_count = len(ALPHA_GRID) * len(L1_RATIO_GRID)
        actual_count = len(children)
        
        print(f"  Expected child models: {expected_count}")
        print(f"  Found child models: {actual_count}")
        
        if actual_count == 0:
            print(f"  ⚠ No child models found!")
            print(f"    This could mean save_all_grid_models=False or children weren't saved")
            
            # Check if any models exist with similar timestamp
            recent = await conn.fetch(
                """SELECT id, parent_model_id, is_grid_member, created_at 
                   FROM models 
                   WHERE created_at > (SELECT created_at FROM models WHERE id = $1)
                   ORDER BY created_at LIMIT 20""",
                parent_model_id
            )
            if recent:
                print(f"  Found {len(recent)} models created after parent:")
                for r in recent[:5]:
                    print(f"    - {r['id'][:8]}... parent_model_id={r['parent_model_id']}, is_grid_member={r['is_grid_member']}")
            return False
        
        # Verify each child
        all_valid = True
        for i, child in enumerate(children):
            print(f"\n  Child {i+1}/{actual_count}:")
            print(f"    ID: {child['id'][:16]}...")
            print(f"    Status: {child['status']}")
            print(f"    parent_model_id: {child['parent_model_id']}")
            print(f"    is_grid_member: {child['is_grid_member']}")
            
            # Verify parent_model_id is correct
            if child['parent_model_id'] == parent_model_id:
                print(f"    ✓ parent_model_id correctly set")
            else:
                print(f"    ✗ parent_model_id MISMATCH!")
                all_valid = False
            
            # Check hyperparameters
            hyperparams = child['hyperparameters']
            if hyperparams:
                if isinstance(hyperparams, str):
                    hyperparams = json.loads(hyperparams)
                alpha = hyperparams.get('alpha')
                l1_ratio = hyperparams.get('l1_ratio')
                print(f"    Hyperparameters: alpha={alpha}, l1_ratio={l1_ratio}")
        
        if actual_count == expected_count:
            print(f"\n  ✓ Correct number of child models created")
        else:
            print(f"\n  ⚠ Expected {expected_count} children, got {actual_count}")
            all_valid = False
        
        if all_valid:
            print(f"  ✓ All child models have correct parent_model_id")
        
        return all_valid
        
    finally:
        await conn.close()


async def main():
    print("\n" + "=" * 60)
    print("   ElasticNet Grid Search Verification")
    print("=" * 60)
    print(f"\nTraining Service: {TRAINING_SERVICE_URL}")
    print(f"Database: {POSTGRES_URL.split('@')[1] if '@' in POSTGRES_URL else POSTGRES_URL}")
    
    # Step 1: Submit training job
    model_id = await submit_training_job(save_all_grid_models=True)
    if not model_id:
        print("\n❌ VERIFICATION FAILED: Could not submit training job")
        return False
    
    # Step 2: Wait for completion
    model = await poll_for_completion(model_id, timeout=180)
    if not model or model.get("status") != "completed":
        print("\n❌ VERIFICATION FAILED: Training did not complete successfully")
        return False
    
    # Step 3: Verify parent model
    parent_valid = await verify_model(model_id)
    if not parent_valid:
        print("\n❌ VERIFICATION FAILED: Parent model validation failed")
        return False
    
    # Step 4: Verify child models
    children_valid = await verify_child_models(model_id)
    
    # Summary
    print("\n" + "=" * 60)
    print("   VERIFICATION SUMMARY")
    print("=" * 60)
    
    if parent_valid and children_valid:
        print("\n✅ ALL VERIFICATIONS PASSED!")
        print("   - Training job completed successfully")
        print("   - Parent model has correct status and metrics")
        print("   - Child models created with correct parent_model_id")
        print(f"\n   Parent Model ID: {model_id}")
        return True
    else:
        print("\n❌ SOME VERIFICATIONS FAILED")
        print(f"   - Parent model valid: {parent_valid}")
        print(f"   - Child models valid: {children_valid}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

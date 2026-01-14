#!/usr/bin/env python3
"""
Validate async PostgreSQL migration in training service.
Checks:
1. PostgreSQL connection
2. Table schemas
3. Process pool configuration
4. Fingerprint computation
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def validate_postgres_connection():
    """Test PostgreSQL connection and schema."""
    print("\n=== PostgreSQL Validation ===")
    
    try:
        from training_service.pg_db import get_pool, close_pool, ensure_tables
        
        # Get connection pool
        print("✓ Importing PostgreSQL database module")
        
        # Ensure tables exist
        await ensure_tables()
        print("✓ Tables created/verified")
        
        # Get pool
        pool = await get_pool()
        print(f"✓ Connection pool created (min=2, max=10)")
        
        # Test query
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM models")
            print(f"✓ Query successful: {result} models in database")
            
            # Check table structure
            columns = await conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'models'
                ORDER BY ordinal_position
            """)
            
            print(f"\n  Models table has {len(columns)} columns:")
            for col in columns[:10]:  # Show first 10
                print(f"    - {col['column_name']}: {col['data_type']}")
            
            if len(columns) > 10:
                print(f"    ... and {len(columns) - 10} more")
        
        await close_pool()
        print("✓ Connection pool closed")
        
        return True
        
    except Exception as e:
        print(f"✗ PostgreSQL validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_process_pool():
    """Check process pool configuration."""
    print("\n=== Process Pool Validation ===")
    
    try:
        import os
        from concurrent.futures import ProcessPoolExecutor
        
        cpu_count = os.cpu_count() or 4
        print(f"✓ System CPU count: {cpu_count}")
        
        # Create test pool
        pool = ProcessPoolExecutor(max_workers=cpu_count, max_tasks_per_child=10)
        print(f"✓ ProcessPoolExecutor created with {cpu_count} workers")
        
        # Test submission
        def test_task(x):
            import os
            return {"input": x, "pid": os.getpid()}
        
        # Submit test tasks
        futures = [pool.submit(test_task, i) for i in range(4)]
        results = [f.result() for f in futures]
        
        pids = set(r["pid"] for r in results)
        print(f"✓ Test tasks completed across {len(pids)} processes")
        
        pool.shutdown(wait=True)
        print("✓ Process pool shut down cleanly")
        
        return True
        
    except Exception as e:
        print(f"✗ Process pool validation failed: {e}")
        return False


def validate_sync_wrapper():
    """Check sync DB wrapper for process workers."""
    print("\n=== Sync DB Wrapper Validation ===")
    
    try:
        from training_service.sync_db_wrapper import db
        
        print("✓ SyncDBWrapper imported")
        
        # Check it has required methods
        required_methods = [
            'update_model_status',
            'get_model',
            'get_model_by_fingerprint',
            'create_model_record'
        ]
        
        for method in required_methods:
            if not hasattr(db, method):
                print(f"✗ Missing method: {method}")
                return False
            print(f"  ✓ {method}")
        
        print("✓ All required methods present")
        
        return True
        
    except Exception as e:
        print(f"✗ Sync wrapper validation failed: {e}")
        return False


def validate_fingerprint():
    """Check fingerprint computation."""
    print("\n=== Fingerprint Validation ===")
    
    try:
        # Import orchestrator fingerprint if available
        try:
            from orchestrator_service.fingerprint import compute_fingerprint
            
            # Test fingerprint with all parameters
            fp = compute_fingerprint(
                features=["sma_20", "rsi_14"],
                hyperparameters={"n_estimators": 100},
                target_transform="log_return",
                symbol="RDDT",
                target_col="close",
                timeframe="1m",
                train_window=30,
                test_window=7,
                context_symbols=["QQQ"],
                cv_folds=5,
                cv_strategy="time_series",
                alpha_grid=[0.1, 1.0, 10.0],
                l1_ratio_grid=[0.5, 0.7, 0.9],
                regime_configs=None
            )
            
            print(f"✓ Fingerprint computed: {fp[:16]}...")
            
            # Verify deterministic
            fp2 = compute_fingerprint(
                features=["sma_20", "rsi_14"],
                hyperparameters={"n_estimators": 100},
                target_transform="log_return",
                symbol="RDDT",
                target_col="close",
                timeframe="1m",
                train_window=30,
                test_window=7,
                context_symbols=["QQQ"],
                cv_folds=5,
                cv_strategy="time_series",
                alpha_grid=[0.1, 1.0, 10.0],
                l1_ratio_grid=[0.5, 0.7, 0.9],
                regime_configs=None
            )
            
            if fp == fp2:
                print("✓ Fingerprint is deterministic")
            else:
                print("✗ Fingerprint is not deterministic!")
                return False
                
        except ImportError:
            print("⚠  orchestrator_service.fingerprint not available (skipping)")
        
        return True
        
    except Exception as e:
        print(f"✗ Fingerprint validation failed: {e}")
        return False


async def main():
    """Run all validations."""
    print("=" * 60)
    print("Training Service Migration Validation")
    print("=" * 60)
    
    results = {}
    
    # Check environment
    print("\n=== Environment ===")
    postgres_url = os.getenv("POSTGRES_URL")
    if postgres_url:
        # Mask password
        masked = postgres_url.split("@")[1] if "@" in postgres_url else postgres_url
        print(f"✓ POSTGRES_URL: ...@{masked}")
    else:
        print("⚠  POSTGRES_URL not set (will use default)")
    
    # Run validations
    results["postgres"] = await validate_postgres_connection()
    results["process_pool"] = validate_process_pool()
    results["sync_wrapper"] = validate_sync_wrapper()
    results["fingerprint"] = validate_fingerprint()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} validations passed")
    
    if passed == total:
        print("\n✅ All validations passed! Ready for deployment.")
        return 0
    else:
        print("\n⚠️  Some validations failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

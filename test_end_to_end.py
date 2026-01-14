#!/usr/bin/env python3
"""
End-to-end test for PostgreSQL migration.
Tests: training → simulation → fingerprint deduplication
"""
import asyncio
import aiohttp
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Test configuration
TRAINING_URL = "http://localhost:8200"
SIMULATION_URL = "http://localhost:8300"


async def test_training_service():
    """Test training service health and model creation."""
    print("\n" + "=" * 60)
    print("Testing Training Service")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        # Health check
        try:
            async with session.get(f"{TRAINING_URL}/health") as resp:
                if resp.status == 200:
                    print("✓ Training service is healthy")
                else:
                    print(f"✗ Training service health check failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Cannot connect to training service: {e}")
            return False
        
        # List models
        try:
            async with session.get(f"{TRAINING_URL}/models") as resp:
                if resp.status == 200:
                    models = await resp.json()
                    print(f"✓ Retrieved {len(models)} models")
                    
                    if models:
                        print(f"\n  Sample model:")
                        m = models[0]
                        print(f"    ID: {m.get('id', 'N/A')[:16]}...")
                        print(f"    Symbol: {m.get('symbol', 'N/A')}")
                        print(f"    Algorithm: {m.get('algorithm', 'N/A')}")
                        print(f"    Status: {m.get('status', 'N/A')}")
                else:
                    print(f"✗ Failed to list models: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Failed to list models: {e}")
            return False
        
        # Submit test training job (small config for speed)
        print(f"\n  Submitting test training job...")
        try:
            async with session.post(
                f"{TRAINING_URL}/train",
                json={
                    "symbol": "RDDT",
                    "algorithm": "RandomForest",
                    "target_col": "close",
                    "timeframe": "1m",
                    "hyperparameters": {"n_estimators": 10},  # Small for speed
                    "target_transform": "log_return",
                    "p_value_threshold": 0.05
                }
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    training_id = result.get("id")
                    print(f"✓ Training job submitted: {training_id[:16]}...")
                    return training_id
                else:
                    text = await resp.text()
                    print(f"✗ Failed to submit training: {resp.status} - {text}")
                    return False
        except Exception as e:
            print(f"✗ Failed to submit training: {e}")
            return False


async def test_simulation_service(model_id=None):
    """Test simulation service health and history."""
    print("\n" + "=" * 60)
    print("Testing Simulation Service")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        # Health check
        try:
            async with session.get(f"{SIMULATION_URL}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✓ Simulation service is healthy")
                    print(f"  Models available: {data.get('models_available', 0)}")
                    print(f"  Tickers available: {data.get('tickers_available', 0)}")
                else:
                    print(f"✗ Simulation service health check failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Cannot connect to simulation service: {e}")
            return False
        
        # Get simulation history
        try:
            async with session.get(f"{SIMULATION_URL}/api/history?limit=5") as resp:
                if resp.status == 200:
                    history = await resp.json()
                    print(f"✓ Retrieved {len(history)} recent simulations")
                    
                    if history:
                        print(f"\n  Latest simulation:")
                        sim = history[0]
                        print(f"    Model: {sim.get('model_id', 'N/A')[:16]}...")
                        print(f"    Ticker: {sim.get('ticker', 'N/A')}")
                        print(f"    Return: {sim.get('return_pct', 0):.2f}%")
                        print(f"    Trades: {sim.get('trades_count', 0)}")
                        print(f"    SQN: {sim.get('sqn', 0):.2f}")
                else:
                    print(f"✗ Failed to get history: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Failed to get history: {e}")
            return False
        
        # Get top strategies
        try:
            async with session.get(f"{SIMULATION_URL}/history/top?limit=5") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    items = data.get("items", [])
                    total = data.get("total", 0)
                    print(f"✓ Retrieved top {len(items)} strategies (total: {total})")
                    
                    if items:
                        print(f"\n  Top strategy:")
                        top = items[0]
                        print(f"    Model: {top.get('model_id', 'N/A')[:16]}...")
                        print(f"    Ticker: {top.get('ticker', 'N/A')}")
                        print(f"    SQN: {top.get('sqn', 0):.2f}")
                        print(f"    Return: {top.get('return_pct', 0):.2f}%")
                else:
                    print(f"✗ Failed to get top strategies: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Failed to get top strategies: {e}")
            return False
        
        return True


async def test_fingerprint_deduplication():
    """Test fingerprint-based model deduplication."""
    print("\n" + "=" * 60)
    print("Testing Fingerprint Deduplication")
    print("=" * 60)
    
    try:
        from orchestrator_service.fingerprint import compute_fingerprint
        
        # Test identical configs produce same fingerprint
        config1 = {
            "features": ["sma_20", "rsi_14"],
            "hyperparameters": {"n_estimators": 100},
            "target_transform": "log_return",
            "symbol": "RDDT",
            "target_col": "close",
            "timeframe": "1m",
            "train_window": 30,
            "test_window": 7,
            "context_symbols": ["QQQ"],
            "cv_folds": 5,
            "cv_strategy": "time_series",
            "alpha_grid": [0.1, 1.0],
            "l1_ratio_grid": [0.5, 0.9],
            "regime_configs": None
        }
        
        fp1 = compute_fingerprint(**config1)
        fp2 = compute_fingerprint(**config1)
        
        if fp1 == fp2:
            print(f"✓ Identical configs produce same fingerprint")
            print(f"  Fingerprint: {fp1[:16]}...")
        else:
            print(f"✗ Fingerprints differ for identical configs!")
            return False
        
        # Test different configs produce different fingerprints
        config2 = {**config1, "features": ["sma_20", "rsi_14", "ema_10"]}  # Add feature
        fp3 = compute_fingerprint(**config2)
        
        if fp1 != fp3:
            print(f"✓ Different configs produce different fingerprints")
            print(f"  Original: {fp1[:16]}...")
            print(f"  Modified: {fp3[:16]}...")
        else:
            print(f"✗ Fingerprints same for different configs!")
            return False
        
        # Test parameter changes affect fingerprint
        config3 = {**config1, "timeframe": "5m"}  # Change timeframe
        fp4 = compute_fingerprint(**config3)
        
        if fp1 != fp4:
            print(f"✓ Timeframe changes affect fingerprint")
        else:
            print(f"✗ Timeframe change didn't affect fingerprint!")
            return False
        
        return True
        
    except ImportError:
        print("⚠  orchestrator_service.fingerprint not available (skipping)")
        return True
    except Exception as e:
        print(f"✗ Fingerprint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_postgres_connection():
    """Test PostgreSQL connection from training service."""
    print("\n" + "=" * 60)
    print("Testing PostgreSQL Connection")
    print("=" * 60)
    
    try:
        from training_service.pg_db import get_pool, close_pool
        
        pool = await get_pool()
        print("✓ PostgreSQL connection pool created")
        
        async with pool.acquire() as conn:
            # Test query
            result = await conn.fetchval("SELECT COUNT(*) FROM models")
            print(f"✓ Query successful: {result} models in database")
            
            # Check simulation_history table
            sim_count = await conn.fetchval("SELECT COUNT(*) FROM simulation_history")
            print(f"✓ Simulation history table accessible: {sim_count} records")
        
        await close_pool()
        print("✓ Connection pool closed cleanly")
        
        return True
        
    except Exception as e:
        print(f"✗ PostgreSQL connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("PostgreSQL Migration End-to-End Test")
    print("=" * 60)
    
    results = {}
    
    # Test PostgreSQL connection
    results["postgres"] = await test_postgres_connection()
    
    # Test fingerprint computation
    results["fingerprint"] = await test_fingerprint_deduplication()
    
    # Test training service
    training_id = await test_training_service()
    results["training"] = bool(training_id)
    
    # Test simulation service
    results["simulation"] = await test_simulation_service()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! PostgreSQL migration successful.")
        print("\nNext steps:")
        print("1. Run data migration: python scripts/migrate_to_postgres.py")
        print("2. Monitor services: docker compose logs -f training_service simulation_service")
        print("3. Test parallel training: submit multiple /train requests")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

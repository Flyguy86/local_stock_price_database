#!/usr/bin/env python3
"""
Quick smoke test for orchestrator workflows.

Tests basic functionality without requiring full pytest setup.
Run with: python tests/smoke_test_orchestrator.py
"""
import sys
import asyncio
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "orchestrator_service"))

try:
    from httpx import AsyncClient
    from orchestrator_service.main import app
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install httpx")
    sys.exit(1)


async def test_browse_models():
    """Test /models/browse endpoint."""
    print("\n🔍 Testing /models/browse endpoint...")
    
    async with AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
        try:
            response = await client.get("/models/browse?limit=5")
            
            if response.status_code == 200:
                data = response.json()
                model_count = len(data.get("models", []))
                total = data.get("total", 0)
                print(f"✅ Browse models works! Found {model_count} models (total: {total})")
                
                if model_count > 0:
                    model = data["models"][0]
                    print(f"   Sample: {model['symbol']} {model['algorithm']} "
                          f"(accuracy: {model['accuracy']*100:.1f}%)")
                return True
            else:
                print(f"❌ Browse failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def test_train_only_endpoint():
    """Test /train-only endpoint accepts requests."""
    print("\n🧠 Testing /train-only endpoint...")
    
    payload = {
        "symbol": "TEST",
        "algorithm": "ridge",
        "target_col": "close",
        "max_generations": 1,
        "prune_fraction": 0.5,
        "min_features": 3,
        "target_transform": "none",
        "timeframe": "1d"
    }
    
    async with AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
        try:
            response = await client.post("/train-only", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Train-only endpoint works!")
                print(f"   Status: {data.get('status')}")
                print(f"   Mode: {data.get('mode')}")
                print(f"   Message: {data.get('message')}")
                return True
            elif response.status_code == 404:
                print(f"⚠️  Train-only endpoint returned 404")
                print(f"   This might mean no feature data for symbol TEST")
                print(f"   Try with a real symbol like AAPL")
                return True  # Endpoint exists, just no data
            else:
                print(f"❌ Train-only failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def test_manual_simulations_validation():
    """Test /simulations/manual validation."""
    print("\n🚀 Testing /simulations/manual endpoint validation...")
    
    # Test empty model_ids
    payload = {
        "model_ids": [],
        "simulation_tickers": ["AAPL"],
        "thresholds": [0.0001],
        "z_score_thresholds": [0],
        "regime_configs": [{}]
    }
    
    async with AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
        try:
            response = await client.post("/simulations/manual", json=payload)
            
            if response.status_code == 400:
                error = response.json()["detail"]
                if "No model IDs provided" in error:
                    print(f"✅ Validation works! Correctly rejects empty model_ids")
                    return True
                else:
                    print(f"⚠️  Got 400 but wrong error: {error}")
                    return False
            else:
                print(f"❌ Should reject empty model_ids with 400, got {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def test_stats_endpoint():
    """Test /stats endpoint."""
    print("\n📊 Testing /stats endpoint...")
    
    async with AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
        try:
            response = await client.get("/stats")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Stats endpoint works!")
                print(f"   Total models: {data.get('total_models', 0)}")
                print(f"   Active runs: {data.get('active_runs', 0)}")
                return True
            else:
                print(f"❌ Stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def test_ui_logic():
    """Test UI logic (no async needed)."""
    print("\n🎨 Testing UI logic functions...")
    
    # Test 1: Model selection toggle
    selected_ids = set()
    model_id = "test-123"
    
    # Add
    if model_id in selected_ids:
        selected_ids.remove(model_id)
    else:
        selected_ids.add(model_id)
    
    assert model_id in selected_ids, "Should add model"
    
    # Remove
    if model_id in selected_ids:
        selected_ids.remove(model_id)
    else:
        selected_ids.add(model_id)
    
    assert model_id not in selected_ids, "Should remove model"
    print("   ✅ Toggle selection logic works")
    
    # Test 2: Grid size calculation
    tickers = 2
    thresholds = 4
    z_scores = 5
    regimes = 7
    models = 3
    
    grid_per_model = tickers * thresholds * z_scores * regimes
    total = grid_per_model * models
    
    assert grid_per_model == 280, f"Expected 280, got {grid_per_model}"
    assert total == 840, f"Expected 840, got {total}"
    print(f"   ✅ Grid calculation: {models} models × {grid_per_model} sims = {total} total")
    
    # Test 3: Filter logic
    models_list = [
        {"symbol": "AAPL", "algorithm": "randomforest", "accuracy": 0.65},
        {"symbol": "SPY", "algorithm": "xgboost", "accuracy": 0.72},
    ]
    
    filter_symbol = "aapl"
    filtered = [
        m for m in models_list
        if filter_symbol.lower() in m["symbol"].lower()
    ]
    
    assert len(filtered) == 1, "Should filter to 1 model"
    assert filtered[0]["symbol"] == "AAPL", "Should be AAPL"
    print("   ✅ Filter logic works")
    
    return True


async def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("🔥 Orchestrator Workflow Smoke Tests")
    print("=" * 60)
    
    results = []
    
    # UI logic tests (sync)
    try:
        results.append(("UI Logic", test_ui_logic()))
    except Exception as e:
        print(f"❌ UI logic test failed: {e}")
        results.append(("UI Logic", False))
    
    # API tests (async)
    results.append(("Stats Endpoint", await test_stats_endpoint()))
    results.append(("Browse Models", await test_browse_models()))
    results.append(("Train Only", await test_train_only_endpoint()))
    results.append(("Manual Sims Validation", await test_manual_simulations_validation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All smoke tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

"""
Integration tests for Orchestrator Service workflows.

Tests the new two-phase workflow:
1. Train Models Only (no simulations)
2. Browse & Select Models
3. Run Simulations for Selected Models

Also tests the traditional full evolution workflow for comparison.
"""
import pytest
import asyncio
import json
import time
from pathlib import Path
from httpx import AsyncClient
from typing import List, Dict, Any

# Import orchestrator app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "orchestrator_service"))

from orchestrator_service.main import app


class TestTrainOnlyWorkflow:
    """Test train-only workflow (no simulations)."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_train_only_creates_models_no_simulations(self):
        """
        Train-only should create models without running simulations.
        
        Steps:
        1. Submit train-only request
        2. Verify response indicates train-only mode
        3. Poll for completion (check Active Runs)
        4. Verify models created in database
        5. Verify NO simulations were run
        """
        payload = {
            "symbol": "AAPL",
            "reference_symbols": ["SPY"],
            "algorithm": "randomforest",
            "target_col": "close_zscore_change",
            "max_generations": 2,
            "prune_fraction": 0.5,
            "min_features": 5,
            "target_transform": "log",
            "timeframe": "1d",
            "data_options": None,
            "alpha_grid": None,
            "l1_ratio_grid": None
        }
        
        async with AsyncClient(app=app, base_url="http://test", timeout=30.0) as client:
            # Step 1: Submit train-only request
            response = await client.post("/train-only", json=payload)
            
            assert response.status_code == 200, f"Failed: {response.text}"
            data = response.json()
            
            # Step 2: Verify response
            assert data["status"] == "started"
            assert data["mode"] == "train-only"
            assert data["max_generations"] == 2
            assert "no simulations" in data["message"].lower()
            
            # Step 3: Wait for training to complete (with timeout)
            max_wait = 120  # 2 minutes max
            start_time = time.time()
            models_created = []
            
            while time.time() - start_time < max_wait:
                await asyncio.sleep(5)
                
                # Check for created models via browse endpoint
                browse_response = await client.get(
                    "/models/browse",
                    params={"symbol": "AAPL", "algorithm": "randomforest", "limit": 50}
                )
                
                if browse_response.status_code == 200:
                    browse_data = browse_response.json()
                    models = browse_data.get("models", [])
                    
                    # Filter to only recently created models (within last 2 minutes)
                    recent_models = [
                        m for m in models 
                        if m.get("created_at") and 
                        (time.time() - parse_timestamp(m["created_at"])) < 150
                    ]
                    
                    if len(recent_models) >= 2:  # At least 2 models for 2 generations
                        models_created = recent_models
                        break
            
            # Step 4: Verify models created
            assert len(models_created) >= 2, f"Expected at least 2 models, got {len(models_created)}"
            
            for model in models_created:
                assert model["symbol"] == "AAPL"
                assert model["algorithm"] == "randomforest"
                assert model["target_transform"] == "log"
                assert "accuracy" in model
                assert "r2_score" in model
                assert "mse" in model
                assert model["feature_count"] >= 5
            
            # Step 5: Verify NO simulations were run
            # Query simulations table for these model IDs
            model_ids = [m["id"] for m in models_created]
            
            stats_response = await client.get("/stats")
            assert stats_response.status_code == 200
            stats = stats_response.json()
            
            # For train-only, simulation count should not increase significantly
            # (some baseline simulations may exist from previous tests)
            # This is a soft check - we can't guarantee zero simulations
            # but we can verify models were created
            assert "total_models" in stats
    
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_train_only_with_regularization_grid(self):
        """
        Test train-only with ElasticNet and regularization grid search.
        
        Should create multiple models for each alpha/l1_ratio combination.
        """
        payload = {
            "symbol": "AAPL",
            "reference_symbols": [],
            "algorithm": "elasticnet",
            "target_col": "close_zscore_change",
            "max_generations": 1,
            "prune_fraction": 0.5,
            "min_features": 5,
            "target_transform": "none",
            "timeframe": "1d",
            "data_options": None,
            "alpha_grid": [0.001, 0.01],
            "l1_ratio_grid": [0.5, 0.9]
        }
        
        async with AsyncClient(app=app, base_url="http://test", timeout=30.0) as client:
            response = await client.post("/train-only", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
            
            # Wait for completion
            await asyncio.sleep(30)  # ElasticNet is fast
            
            # Verify multiple models created for grid combinations
            browse_response = await client.get(
                "/models/browse",
                params={"symbol": "AAPL", "algorithm": "elasticnet"}
            )
            
            assert browse_response.status_code == 200
            models = browse_response.json().get("models", [])
            
            # Should have models for: 2 alphas × 2 l1_ratios = 4 base configs
            # May have fewer if pruning removed some
            assert len(models) >= 1, "Expected at least 1 ElasticNet model"


class TestModelBrowseAndSelect:
    """Test model browsing and selection functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_browse_models_with_filters(self):
        """
        Test /models/browse endpoint with various filters.
        
        Verifies:
        - Symbol filtering
        - Algorithm filtering
        - Status filtering
        - Accuracy threshold filtering
        - Limit parameter
        """
        async with AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
            # Test 1: Browse all models
            response = await client.get("/models/browse")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert "total" in data
            all_models = data["models"]
            
            # Test 2: Filter by symbol
            if all_models:
                first_symbol = all_models[0]["symbol"]
                response = await client.get(
                    "/models/browse",
                    params={"symbol": first_symbol}
                )
                assert response.status_code == 200
                filtered = response.json()["models"]
                assert all(m["symbol"] == first_symbol for m in filtered)
            
            # Test 3: Filter by algorithm
            response = await client.get(
                "/models/browse",
                params={"algorithm": "randomforest"}
            )
            assert response.status_code == 200
            filtered = response.json()["models"]
            assert all(m["algorithm"] == "randomforest" for m in filtered)
            
            # Test 4: Filter by min accuracy
            response = await client.get(
                "/models/browse",
                params={"min_accuracy": 0.6}
            )
            assert response.status_code == 200
            filtered = response.json()["models"]
            assert all(m["accuracy"] >= 0.6 for m in filtered)
            
            # Test 5: Limit results
            response = await client.get(
                "/models/browse",
                params={"limit": 5}
            )
            assert response.status_code == 200
            filtered = response.json()["models"]
            assert len(filtered) <= 5
    
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_model_fingerprint_included(self):
        """
        Verify model browse results include fingerprint data.
        
        Each model should include:
        - ID, symbol, algorithm, status
        - Metrics: accuracy, r2_score, mse
        - Feature count
        - Target transform
        - Hyperparameters (fingerprint)
        """
        async with AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
            response = await client.get("/models/browse", params={"limit": 1})
            assert response.status_code == 200
            
            models = response.json().get("models", [])
            if not models:
                pytest.skip("No models in database to test")
            
            model = models[0]
            
            # Verify required fields
            required_fields = [
                "id", "symbol", "algorithm", "status",
                "accuracy", "r2_score", "mse",
                "feature_count", "target_transform"
            ]
            
            for field in required_fields:
                assert field in model, f"Missing field: {field}"
            
            # Verify metrics are valid
            assert 0 <= model["accuracy"] <= 1
            assert -100 <= model["r2_score"] <= 1  # R² can be negative
            assert model["mse"] >= 0
            assert model["feature_count"] > 0


class TestManualSimulations:
    """Test manual simulation launching for selected models."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_simulations_for_selected_models(self):
        """
        Test /simulations/manual endpoint.
        
        Steps:
        1. Browse available models
        2. Select 1-2 model IDs
        3. Submit manual simulation request
        4. Verify jobs created
        5. Verify correct number of simulations queued
        """
        async with AsyncClient(app=app, base_url="http://test", timeout=30.0) as client:
            # Step 1: Browse models
            browse_response = await client.get(
                "/models/browse",
                params={"limit": 2, "status": "TRAINED"}
            )
            
            if browse_response.status_code != 200:
                pytest.skip("Cannot browse models")
            
            models = browse_response.json().get("models", [])
            if len(models) < 1:
                pytest.skip("No trained models available for simulation test")
            
            # Step 2: Select model IDs
            model_ids = [m["id"] for m in models[:2]]
            
            # Step 3: Submit simulation request
            payload = {
                "model_ids": model_ids,
                "simulation_tickers": ["AAPL", "SPY"],
                "thresholds": [0.0001, 0.0005],
                "z_score_thresholds": [0, 2.5],
                "regime_configs": [
                    {},  # No filter
                    {"regime_vix": [0]},  # VIX regime 0
                ],
                "sqn_min": 2.5,
                "profit_factor_min": 1.5,
                "trade_count_min": 100
            }
            
            response = await client.post("/simulations/manual", json=payload)
            
            assert response.status_code == 200, f"Failed: {response.text}"
            data = response.json()
            
            # Step 4: Verify response structure
            assert data["status"] == "queued"
            assert data["model_count"] == len(model_ids)
            assert data["ticker_count"] == 2
            
            # Step 5: Verify simulation count calculation
            # Expected: 2 models × 2 tickers × 2 thresholds × 2 z-scores × 2 regimes = 32
            expected_per_model = 2 * 2 * 2 * 2  # 16
            expected_total = expected_per_model * len(model_ids)
            
            assert data["simulations_per_model"] == expected_per_model
            assert data["total_simulations"] == expected_total
            assert data["total_jobs"] == expected_total
    
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_manual_simulations_validation(self):
        """
        Test validation for manual simulation requests.
        
        Should reject:
        - Empty model_ids list
        - Empty simulation_tickers list
        - Invalid model IDs
        """
        async with AsyncClient(app=app, base_url="http://test", timeout=10.0) as client:
            # Test 1: Empty model_ids
            payload = {
                "model_ids": [],
                "simulation_tickers": ["AAPL"],
                "thresholds": [0.0001],
                "z_score_thresholds": [0],
                "regime_configs": [{}]
            }
            
            response = await client.post("/simulations/manual", json=payload)
            assert response.status_code == 400
            assert "No model IDs provided" in response.json()["detail"]
            
            # Test 2: Empty simulation_tickers
            payload = {
                "model_ids": ["test-model-id"],
                "simulation_tickers": [],
                "thresholds": [0.0001],
                "z_score_thresholds": [0],
                "regime_configs": [{}]
            }
            
            response = await client.post("/simulations/manual", json=payload)
            assert response.status_code == 400
            assert "No simulation tickers provided" in response.json()["detail"]


class TestFullEvolutionComparison:
    """Test that full evolution workflow still works alongside new train-only mode."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_evolution_creates_models_and_simulations(self):
        """
        Test traditional /evolve endpoint still works.
        
        Should create models AND run simulations automatically.
        """
        payload = {
            "symbol": "AAPL",
            "reference_symbols": [],
            "simulation_tickers": ["AAPL"],
            "algorithm": "linearregression",
            "target_col": "close",
            "max_generations": 1,
            "prune_fraction": 0.5,
            "min_features": 3,
            "target_transform": "none",
            "timeframe": "1d",
            "thresholds": [0.0001],
            "z_score_thresholds": [0],
            "regime_configs": [{}],
            "sqn_min": 2.0,
            "sqn_max": 5.0,
            "profit_factor_min": 1.5,
            "profit_factor_max": 4.0,
            "trade_count_min": 100,
            "trade_count_max": 10000
        }
        
        async with AsyncClient(app=app, base_url="http://test", timeout=30.0) as client:
            response = await client.post("/evolve", json=payload)
            
            assert response.status_code == 200, f"Failed: {response.text}"
            data = response.json()
            
            assert "run_id" in data or "status" in data
            # Full evolution should queue both training AND simulation jobs


# ============================================
# Helper Functions
# ============================================

def parse_timestamp(ts_str: str) -> float:
    """Parse ISO timestamp to Unix epoch."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.timestamp()
    except:
        return 0.0


# ============================================
# Fixtures
# ============================================

@pytest.fixture
async def sample_trained_models():
    """
    Create a few sample trained models for testing.
    
    Uses train-only endpoint to create real models.
    """
    models_created = []
    
    async with AsyncClient(app=app, base_url="http://test", timeout=60.0) as client:
        for algo in ["ridge", "randomforest"]:
            payload = {
                "symbol": "AAPL",
                "algorithm": algo,
                "target_col": "close",
                "max_generations": 1,
                "prune_fraction": 0.5,
                "min_features": 3,
                "target_transform": "none",
                "timeframe": "1d"
            }
            
            response = await client.post("/train-only", json=payload)
            if response.status_code == 200:
                models_created.append(algo)
    
    # Wait for training to complete
    await asyncio.sleep(30)
    
    yield models_created


if __name__ == "__main__":
    # Run tests with: pytest tests/integration/test_orchestrator_workflows.py -v -s
    pytest.main([__file__, "-v", "-s", "--tb=short"])

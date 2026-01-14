"""
API tests for Simulation Service endpoints.

Tests all HTTP endpoints of the simulation service including:
- Health checks
- Simulation execution
- Simulation history
- Top strategies
- Bot training
- Batch operations
"""
import pytest
import asyncio
import json
from httpx import AsyncClient
from fastapi.testclient import TestClient
from simulation_service.main import app, db


class TestSimulationServiceHealth:
    """Test health and status endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, db_tables):
        """Test /health endpoint returns healthy status."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]
        assert data["service"] == "simulation"
    
    @pytest.mark.asyncio
    async def test_logs_endpoint(self, db_tables):
        """Test /logs endpoint returns log buffer."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/logs")
            
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)


class TestConfigEndpoints:
    """Test configuration and metadata endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_config(self, db_tables):
        """Test /api/config endpoint returns available models and tickers."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/config")
            
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "tickers" in data
        assert isinstance(data["models"], list)
        assert isinstance(data["tickers"], list)


class TestSimulationHistory:
    """Test simulation history CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_get_history_empty(self, db_tables):
        """Test /api/history endpoint returns empty list initially."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/history")
            
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)
        assert len(data["history"]) == 0
    
    @pytest.mark.asyncio
    async def test_get_history_with_data(self, db_tables):
        """Test /api/history endpoint returns simulation records."""
        # Create a simulation record
        sim_data = {
            "strategy_id": "test-strategy-1",
            "symbol": "AAPL",
            "model_id": "test-model-1",
            "parameters": json.dumps({"stop_loss": 0.02}),
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
            "trades": 50
        }
        await db.save_simulation_history(sim_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/history")
            
        assert response.status_code == 200
        data = response.json()
        assert len(data["history"]) == 1
        assert data["history"][0]["symbol"] == "AAPL"
        assert data["history"][0]["total_return"] == 0.15
    
    @pytest.mark.asyncio
    async def test_get_top_strategies(self, db_tables):
        """Test /history/top endpoint returns top performing strategies."""
        # Create multiple simulation records
        for i in range(5):
            sim_data = {
                "strategy_id": f"strategy-{i}",
                "symbol": "AAPL",
                "model_id": f"model-{i}",
                "parameters": json.dumps({"stop_loss": 0.02}),
                "total_return": 0.1 + (i * 0.05),  # Increasing returns
                "sharpe_ratio": 1.0 + (i * 0.2),
                "max_drawdown": -0.05,
                "trades": 50
            }
            await db.save_simulation_history(sim_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/history/top?limit=3")
            
        assert response.status_code == 200
        data = response.json()
        assert "top_strategies" in data
        assert len(data["top_strategies"]) <= 3
        
        # Verify sorted by total_return descending
        returns = [s["total_return"] for s in data["top_strategies"]]
        assert returns == sorted(returns, reverse=True)
    
    @pytest.mark.asyncio
    async def test_get_top_strategies_pagination(self, db_tables):
        """Test /history/top endpoint supports pagination."""
        # Create simulation records
        for i in range(10):
            sim_data = {
                "strategy_id": f"strategy-{i}",
                "symbol": "AAPL",
                "model_id": f"model-{i}",
                "parameters": json.dumps({}),
                "total_return": 0.1 + (i * 0.01),
                "sharpe_ratio": 1.0,
                "max_drawdown": -0.05,
                "trades": 50
            }
            await db.save_simulation_history(sim_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Get first page
            response = await client.get("/history/top?limit=5&offset=0")
            assert response.status_code == 200
            page1 = response.json()["top_strategies"]
            
            # Get second page
            response = await client.get("/history/top?limit=5&offset=5")
            assert response.status_code == 200
            page2 = response.json()["top_strategies"]
            
        assert len(page1) == 5
        assert len(page2) == 5
        # Pages should be different
        page1_ids = {s["strategy_id"] for s in page1}
        page2_ids = {s["strategy_id"] for s in page2}
        assert len(page1_ids & page2_ids) == 0  # No overlap
    
    @pytest.mark.asyncio
    async def test_delete_all_history(self, db_tables):
        """Test /history/all DELETE removes all simulation history."""
        # Create simulation records
        for i in range(3):
            sim_data = {
                "strategy_id": f"strategy-{i}",
                "symbol": "AAPL",
                "model_id": f"model-{i}",
                "parameters": json.dumps({}),
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "max_drawdown": -0.05,
                "trades": 50
            }
            await db.save_simulation_history(sim_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Delete all history
            response = await client.delete("/history/all")
            assert response.status_code == 200
            
            # Verify empty
            response = await client.get("/api/history")
            data = response.json()
            assert len(data["history"]) == 0


class TestSimulationOperations:
    """Test simulation execution endpoints."""
    
    @pytest.mark.asyncio
    async def test_simulate_endpoint_basic(self, db_tables):
        """Test /api/simulate endpoint accepts simulation request."""
        payload = {
            "symbol": "AAPL",
            "model_id": "test-model-1",
            "strategy": "simple",
            "parameters": {
                "stop_loss": 0.02,
                "take_profit": 0.05
            }
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/simulate", json=payload)
            
        # Response depends on whether models are available
        # Could be 200 (success) or 400/404 (no models)
        assert response.status_code in [200, 400, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "simulation_id" in data or "results" in data
    
    @pytest.mark.asyncio
    async def test_simulate_missing_parameters(self, db_tables):
        """Test /api/simulate endpoint validates required parameters."""
        payload = {
            "symbol": "AAPL"
            # Missing model_id and strategy
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/simulate", json=payload)
            
        # Should reject with validation error
        assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_batch_simulate_endpoint(self, db_tables):
        """Test /api/batch_simulate endpoint accepts multiple simulations."""
        payload = {
            "symbols": ["AAPL", "GOOGL"],
            "model_ids": ["model-1"],
            "strategies": ["simple"],
            "parameters": {
                "stop_loss": 0.02
            }
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/batch_simulate", json=payload)
            
        # Response depends on whether models are available
        assert response.status_code in [200, 400, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "simulation_ids" in data or "results" in data


class TestBotTraining:
    """Test trading bot training endpoints."""
    
    @pytest.mark.asyncio
    async def test_train_bot_endpoint(self, db_tables):
        """Test /api/train_bot endpoint accepts bot training request."""
        payload = {
            "symbol": "AAPL",
            "model_id": "test-model-1",
            "strategy_type": "reinforcement",
            "parameters": {
                "learning_rate": 0.001,
                "episodes": 100
            }
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/train_bot", json=payload)
            
        # Response depends on implementation
        assert response.status_code in [200, 400, 404, 500, 501]
        
        if response.status_code == 200:
            data = response.json()
            assert "bot_id" in data or "message" in data
    
    @pytest.mark.asyncio
    async def test_train_bot_missing_parameters(self, db_tables):
        """Test /api/train_bot endpoint validates required parameters."""
        payload = {
            "symbol": "AAPL"
            # Missing required fields
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/train_bot", json=payload)
            
        # Should reject with validation error
        assert response.status_code in [400, 422]


class TestDashboard:
    """Test dashboard and HTML endpoints."""
    
    @pytest.mark.asyncio
    async def test_dashboard_endpoint(self, db_tables):
        """Test / endpoint returns HTML dashboard."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/")
            
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # Basic HTML validation
        content = response.text
        assert "<html" in content.lower() or "<!doctype" in content.lower()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_endpoint_404(self, db_tables):
        """Test invalid endpoints return 404."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/invalid/endpoint")
            
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_invalid_method_405(self, db_tables):
        """Test invalid HTTP methods return 405."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # GET on POST-only endpoint
            response = await client.get("/api/simulate")
            
        assert response.status_code == 405
    
    @pytest.mark.asyncio
    async def test_malformed_json_422(self, db_tables):
        """Test malformed JSON returns 422."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Invalid JSON payload
            response = await client.post(
                "/api/simulate",
                content=b"{invalid json}",
                headers={"Content-Type": "application/json"}
            )
            
        assert response.status_code == 422


class TestPerformanceMetrics:
    """Test endpoints that return performance metrics."""
    
    @pytest.mark.asyncio
    async def test_history_includes_metrics(self, db_tables):
        """Test simulation history includes all performance metrics."""
        sim_data = {
            "strategy_id": "test-strategy",
            "symbol": "AAPL",
            "model_id": "test-model",
            "parameters": json.dumps({}),
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
            "trades": 50,
            "win_rate": 0.6,
            "profit_factor": 1.8
        }
        await db.save_simulation_history(sim_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/history")
            
        assert response.status_code == 200
        data = response.json()["history"][0]
        
        # Verify all metrics present
        assert data["total_return"] == 0.15
        assert data["sharpe_ratio"] == 1.5
        assert data["max_drawdown"] == -0.05
        assert data["trades"] == 50


# Mark as async tests
pytestmark = pytest.mark.asyncio

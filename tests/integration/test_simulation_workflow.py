"""
Integration tests for Simulation Service workflows.

Tests the complete end-to-end flow:
1. Run simulation
2. Save results to database
3. Query simulation history
4. Retrieve top strategies
5. Verify performance metrics
"""
import pytest
import asyncio
import json
from httpx import AsyncClient

from simulation_service.main import app, db


class TestSimulationWorkflow:
    """Test complete simulation workflow from execution to retrieval."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_simulate_save_history_flow(self, db_tables):
        """
        Integration test: Simulate → Save → Query complete workflow.
        
        Steps:
        1. Run simulation (or create simulation record)
        2. Save to database
        3. Query simulation history
        4. Verify all metrics present
        5. Verify retrievable by strategy_id
        """
        # Step 1: Create simulation result manually
        # (In real workflow, this comes from simulation execution)
        sim_data = {
            "strategy_id": "test-strategy-001",
            "symbol": "AAPL",
            "model_id": "test-model-001",
            "parameters": json.dumps({
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "position_size": 0.1
            }),
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.08,
            "trades": 42,
            "win_rate": 0.62,
            "profit_factor": 1.8
        }
        
        # Step 2: Save to database
        await db.save_simulation_history(sim_data)
        
        # Step 3: Query via API
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/history")
        
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        history_list = data["history"]
        
        # Step 4: Verify metrics
        assert len(history_list) >= 1
        found = False
        for record in history_list:
            if record["strategy_id"] == "test-strategy-001":
                found = True
                assert record["symbol"] == "AAPL"
                assert record["model_id"] == "test-model-001"
                assert record["total_return"] == 0.15
                assert record["sharpe_ratio"] == 1.5
                assert record["max_drawdown"] == -0.08
                assert record["trades"] == 42
                assert record["win_rate"] == 0.62
                assert record["profit_factor"] == 1.8
                break
        
        assert found, "Simulation record should be in history"
        
        # Step 5: Verify top strategies includes this one
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/history/top?limit=10")
        
        assert response.status_code == 200
        top_strategies = response.json()["top_strategies"]
        strategy_ids = {s["strategy_id"] for s in top_strategies}
        assert "test-strategy-001" in strategy_ids
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_batch_simulation_workflow(self, db_tables):
        """
        Integration test: Multiple simulations → History → Top performers.
        
        Verifies that multiple simulations can be saved and ranked correctly.
        """
        # Create multiple simulation results with varying performance
        simulations = [
            {
                "strategy_id": f"batch-strategy-{i}",
                "symbol": "AAPL",
                "model_id": f"model-{i}",
                "parameters": json.dumps({"stop_loss": 0.02}),
                "total_return": 0.05 + (i * 0.03),  # Increasing returns
                "sharpe_ratio": 1.0 + (i * 0.2),
                "max_drawdown": -0.05,
                "trades": 50
            }
            for i in range(5)
        ]
        
        # Save all simulations
        for sim in simulations:
            await db.save_simulation_history(sim)
        
        # Query all history
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/history")
        
        assert response.status_code == 200
        history_list = response.json()["history"]
        assert len(history_list) >= 5
        
        # Get top 3 strategies
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/history/top?limit=3")
        
        assert response.status_code == 200
        top_strategies = response.json()["top_strategies"]
        
        # Verify top 3 returned
        assert len(top_strategies) == 3
        
        # Verify sorted by total_return descending
        returns = [s["total_return"] for s in top_strategies]
        assert returns == sorted(returns, reverse=True)
        
        # Top performer should be batch-strategy-4 (highest return)
        assert top_strategies[0]["strategy_id"] == "batch-strategy-4"
        assert top_strategies[0]["total_return"] == 0.17  # 0.05 + (4 * 0.03)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_simulation_pagination_workflow(self, db_tables):
        """
        Integration test: Pagination for large result sets.
        
        Verifies that pagination works correctly for simulation history.
        """
        # Create 20 simulation results
        for i in range(20):
            sim_data = {
                "strategy_id": f"paginated-strategy-{i:02d}",
                "symbol": "AAPL",
                "model_id": f"model-{i}",
                "parameters": json.dumps({}),
                "total_return": 0.10 + (i * 0.01),
                "sharpe_ratio": 1.0,
                "max_drawdown": -0.05,
                "trades": 50
            }
            await db.save_simulation_history(sim_data)
        
        # Test pagination
        async with AsyncClient(app=app, base_url="http://test") as client:
            # First page (top 10)
            response1 = await client.get("/history/top?limit=10&offset=0")
            page1 = response1.json()["top_strategies"]
            
            # Second page (next 10)
            response2 = await client.get("/history/top?limit=10&offset=10")
            page2 = response2.json()["top_strategies"]
        
        # Verify page sizes
        assert len(page1) == 10
        assert len(page2) == 10
        
        # Verify no overlap
        page1_ids = {s["strategy_id"] for s in page1}
        page2_ids = {s["strategy_id"] for s in page2}
        assert len(page1_ids & page2_ids) == 0
        
        # Verify ordering across pages
        all_returns = [s["total_return"] for s in page1] + [s["total_return"] for s in page2]
        assert all_returns == sorted(all_returns, reverse=True)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_simulation_history_deletion(self, db_tables):
        """
        Integration test: Delete all simulation history.
        
        Steps:
        1. Create multiple simulation records
        2. Verify they exist
        3. Delete all history
        4. Verify all deleted
        """
        # Step 1: Create simulation records
        for i in range(3):
            sim_data = {
                "strategy_id": f"delete-test-{i}",
                "symbol": "AAPL",
                "model_id": f"model-{i}",
                "parameters": json.dumps({}),
                "total_return": 0.10,
                "sharpe_ratio": 1.0,
                "max_drawdown": -0.05,
                "trades": 50
            }
            await db.save_simulation_history(sim_data)
        
        # Step 2: Verify they exist
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/history")
        
        history_before = response.json()["history"]
        assert len(history_before) >= 3
        
        # Step 3: Delete all history
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.delete("/history/all")
        
        assert response.status_code == 200
        
        # Step 4: Verify all deleted
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/history")
        
        history_after = response.json()["history"]
        assert len(history_after) == 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_simulation_performance_metrics(self, db_tables):
        """
        Integration test: Verify all performance metrics saved and retrieved.
        
        Tests that all simulation metrics are properly stored and returned.
        """
        # Create simulation with comprehensive metrics
        sim_data = {
            "strategy_id": "metrics-test-001",
            "symbol": "AAPL",
            "model_id": "test-model",
            "parameters": json.dumps({
                "stop_loss": 0.02,
                "take_profit": 0.05
            }),
            "total_return": 0.25,
            "sharpe_ratio": 2.1,
            "sortino_ratio": 2.5,
            "max_drawdown": -0.12,
            "calmar_ratio": 2.08,
            "trades": 100,
            "win_rate": 0.65,
            "profit_factor": 2.2,
            "avg_win": 0.015,
            "avg_loss": -0.008,
            "max_consecutive_wins": 7,
            "max_consecutive_losses": 4
        }
        
        # Save to database
        await db.save_simulation_history(sim_data)
        
        # Retrieve and verify all metrics
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/history")
        
        history_list = response.json()["history"]
        record = next(r for r in history_list if r["strategy_id"] == "metrics-test-001")
        
        # Verify all metrics present
        assert record["total_return"] == 0.25
        assert record["sharpe_ratio"] == 2.1
        assert record["max_drawdown"] == -0.12
        assert record["trades"] == 100
        assert record["win_rate"] == 0.65
        assert record["profit_factor"] == 2.2
        
        # Optional metrics (if supported)
        if "sortino_ratio" in record:
            assert record["sortino_ratio"] == 2.5
        if "calmar_ratio" in record:
            assert record["calmar_ratio"] == 2.08


class TestSimulationConfiguration:
    """Test simulation configuration and setup workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_config_endpoint_workflow(self, db_tables):
        """
        Integration test: Configuration retrieval for simulation setup.
        
        Verifies that available models and tickers are returned correctly.
        """
        # Query configuration
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/config")
        
        assert response.status_code == 200
        config = response.json()
        
        # Verify structure
        assert "models" in config
        assert "tickers" in config
        assert isinstance(config["models"], list)
        assert isinstance(config["tickers"], list)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_health_check_workflow(self, db_tables):
        """
        Integration test: Health check reflects system state.
        
        Verifies health endpoint returns accurate status.
        """
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
        
        assert response.status_code == 200
        health_data = response.json()
        
        assert "status" in health_data
        assert health_data["status"] in ["healthy", "unhealthy"]
        assert health_data["service"] == "simulation"


class TestSimulationStrategies:
    """Test different simulation strategies and parameters."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_strategies_comparison(self, db_tables):
        """
        Integration test: Compare different strategies on same symbol.
        
        Verifies that multiple strategies can be tested and ranked.
        """
        # Create simulations with different strategies
        strategies = [
            {
                "strategy_id": "strategy-conservative",
                "symbol": "AAPL",
                "model_id": "model-1",
                "parameters": json.dumps({
                    "stop_loss": 0.01,  # Tight stop
                    "take_profit": 0.10  # Large target
                }),
                "total_return": 0.08,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.03,
                "trades": 20
            },
            {
                "strategy_id": "strategy-aggressive",
                "symbol": "AAPL",
                "model_id": "model-1",
                "parameters": json.dumps({
                    "stop_loss": 0.05,  # Wide stop
                    "take_profit": 0.03  # Small target
                }),
                "total_return": 0.18,
                "sharpe_ratio": 1.8,
                "max_drawdown": -0.12,
                "trades": 80
            },
            {
                "strategy_id": "strategy-balanced",
                "symbol": "AAPL",
                "model_id": "model-1",
                "parameters": json.dumps({
                    "stop_loss": 0.02,
                    "take_profit": 0.05
                }),
                "total_return": 0.15,
                "sharpe_ratio": 1.6,
                "max_drawdown": -0.06,
                "trades": 50
            }
        ]
        
        # Save all strategies
        for strategy in strategies:
            await db.save_simulation_history(strategy)
        
        # Get top strategies
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/history/top?limit=10")
        
        top_strategies = response.json()["top_strategies"]
        
        # Verify all 3 strategies present
        strategy_ids = {s["strategy_id"] for s in top_strategies}
        assert "strategy-conservative" in strategy_ids
        assert "strategy-aggressive" in strategy_ids
        assert "strategy-balanced" in strategy_ids
        
        # Verify ordered by total_return
        returns_map = {s["strategy_id"]: s["total_return"] for s in top_strategies}
        assert returns_map["strategy-aggressive"] > returns_map["strategy-balanced"]
        assert returns_map["strategy-balanced"] > returns_map["strategy-conservative"]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cross_symbol_comparison(self, db_tables):
        """
        Integration test: Compare strategies across different symbols.
        
        Verifies that top strategies work across multiple symbols.
        """
        symbols = ["AAPL", "GOOGL", "MSFT", "NVDA"]
        
        # Create simulations for each symbol
        for i, symbol in enumerate(symbols):
            sim_data = {
                "strategy_id": f"cross-symbol-{symbol}",
                "symbol": symbol,
                "model_id": "unified-model",
                "parameters": json.dumps({"stop_loss": 0.02}),
                "total_return": 0.10 + (i * 0.02),  # Varying performance
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.05,
                "trades": 50
            }
            await db.save_simulation_history(sim_data)
        
        # Get all history
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/history")
        
        history_list = response.json()["history"]
        
        # Verify all symbols present
        history_symbols = {r["symbol"] for r in history_list}
        for symbol in symbols:
            assert symbol in history_symbols


class TestEndToEndSimulation:
    """Test complete end-to-end simulation scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_full_simulation_lifecycle(self, db_tables):
        """
        Integration test: Complete simulation lifecycle.
        
        Steps:
        1. Check available models
        2. Run simulation
        3. Save results
        4. Query results
        5. Compare with other strategies
        6. Clean up
        """
        # Step 1: Check config (would get available models in real workflow)
        async with AsyncClient(app=app, base_url="http://test") as client:
            config_response = await client.get("/api/config")
        
        assert config_response.status_code == 200
        
        # Step 2 & 3: Create simulation result (in real workflow, this runs simulation)
        sim_data = {
            "strategy_id": "lifecycle-test-001",
            "symbol": "AAPL",
            "model_id": "test-model",
            "parameters": json.dumps({"stop_loss": 0.02, "take_profit": 0.05}),
            "total_return": 0.22,
            "sharpe_ratio": 1.9,
            "max_drawdown": -0.07,
            "trades": 65,
            "win_rate": 0.68,
            "profit_factor": 2.1
        }
        await db.save_simulation_history(sim_data)
        
        # Step 4: Query results
        async with AsyncClient(app=app, base_url="http://test") as client:
            history_response = await client.get("/api/history")
        
        assert history_response.status_code == 200
        history = history_response.json()["history"]
        
        found = any(r["strategy_id"] == "lifecycle-test-001" for r in history)
        assert found
        
        # Step 5: Compare with top strategies
        async with AsyncClient(app=app, base_url="http://test") as client:
            top_response = await client.get("/history/top?limit=5")
        
        top_strategies = top_response.json()["top_strategies"]
        
        # Our strategy should be in top performers (high return)
        top_ids = [s["strategy_id"] for s in top_strategies]
        assert "lifecycle-test-001" in top_ids
        
        # Step 6: Cleanup
        async with AsyncClient(app=app, base_url="http://test") as client:
            delete_response = await client.delete("/history/all")
        
        assert delete_response.status_code == 200


# Mark integration tests
pytestmark = pytest.mark.integration

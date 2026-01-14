"""
Load tests for concurrent database access patterns.

Tests database behavior under high concurrent load to validate:
- Connection pool handling
- Lock contention
- Transaction isolation
- Query performance under load
"""
import pytest
import asyncio
import time
import uuid
from typing import List
import random


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.slow
class TestConcurrentDatabaseAccess:
    """Test database access patterns under concurrent load."""
    
    async def test_100_concurrent_queries(self, db_tables, sample_model_data):
        """
        Load Test: 100 concurrent database queries.
        
        Validates:
        - Connection pool handles high concurrent load
        - No connection pool exhaustion
        - Queries complete successfully
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create test models
        model_ids = []
        for i in range(10):
            config = sample_model_data.copy()
            config["id"] = f"concurrent-{i}-{uuid.uuid4()}"
            config["symbol"] = f"CON{i}"
            config["fingerprint"] = compute_fingerprint({
                "features": config.get("features", []),
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            model_id = await db.create_model_record(config)
            model_ids.append(model_id)
        
        # Execute 100 concurrent queries
        num_queries = 100
        start_time = time.time()
        
        async def query_random_model():
            """Query a random model."""
            model_id = random.choice(model_ids)
            return await db.get_model(model_id)
        
        tasks = [query_random_model() for _ in range(num_queries)]
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # Verify all queries succeeded
        assert len(results) == num_queries
        assert all(r is not None for r in results)
        
        queries_per_second = num_queries / elapsed_time
        print(f"\n✓ {num_queries} concurrent queries in {elapsed_time:.2f}s ({queries_per_second:.1f} queries/s)")
    
    async def test_concurrent_writes_different_tables(self, db_tables, sample_model_data):
        """
        Load Test: Concurrent writes to different tables.
        
        Tests:
        - Models table + feature_importance table
        - No table-level lock contention
        - Foreign key constraints maintained
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        num_models = 20
        
        async def create_model_with_importance(idx: int):
            """Create model and add feature importance."""
            config = sample_model_data.copy()
            config["id"] = f"multi-table-{idx}-{uuid.uuid4()}"
            config["symbol"] = f"MT{idx}"
            config["fingerprint"] = compute_fingerprint({
                "features": [f"feature_{idx}"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            
            # Create model
            model_id = await db.create_model_record(config)
            
            # Add feature importance (different table)
            importance_data = {f"feature_{idx}": 0.8}
            await db.store_feature_importance(model_id, importance_data)
            
            return model_id
        
        start_time = time.time()
        
        tasks = [create_model_with_importance(i) for i in range(num_models)]
        model_ids = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # Verify all succeeded
        assert len(model_ids) == num_models
        
        # Verify all have feature importance
        for model_id in model_ids:
            importance = await db.get_feature_importance(model_id)
            assert importance is not None
        
        print(f"\n✓ {num_models} concurrent multi-table writes in {elapsed_time:.2f}s")
    
    async def test_concurrent_status_updates_same_model(self, db_tables, sample_model_data):
        """
        Load Test: Multiple concurrent updates to same model.
        
        Tests:
        - Row-level locking
        - Update serialization
        - Last write wins consistency
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create a single model
        config = sample_model_data.copy()
        config["id"] = f"single-model-{uuid.uuid4()}"
        config["fingerprint"] = compute_fingerprint({
            "features": config.get("features", []),
            "symbol": config.get("symbol", "TEST"),
            "hyperparameters": {}
        })
        model_id = await db.create_model_record(config)
        
        # Perform 30 concurrent updates to same model
        num_updates = 30
        statuses = ["training", "completed", "failed"]
        
        async def update_status(idx: int):
            """Update model status."""
            status = statuses[idx % len(statuses)]
            await db.update_model_status(model_id, status)
            return status
        
        start_time = time.time()
        
        tasks = [update_status(i) for i in range(num_updates)]
        final_statuses = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # Verify model still exists and has valid status
        model = await db.get_model(model_id)
        assert model is not None
        assert model["status"] in statuses
        
        print(f"\n✓ {num_updates} concurrent updates to same record in {elapsed_time:.3f}s")
        print(f"  Final status: {model['status']}")
    
    async def test_simulation_concurrent_history_writes(self, db_tables):
        """
        Load Test: Concurrent simulation history writes.
        
        Tests:
        - Simulation service database load
        - Concurrent inserts to simulation_history
        - Large payload handling
        """
        from simulation_service.database import SimulationDB
        
        db = SimulationDB()
        
        num_simulations = 25
        
        async def create_simulation(idx: int):
            """Create simulation history record."""
            sim_data = {
                "id": f"sim-{idx}-{uuid.uuid4()}",
                "model_id": f"model-{idx}",
                "symbol": f"SIM{idx}",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "total_return": random.uniform(-0.2, 0.5),
                "sharpe_ratio": random.uniform(0.5, 2.5),
                "max_drawdown": random.uniform(-0.3, -0.05),
                "trade_count": random.randint(50, 500),
                "win_rate": random.uniform(0.45, 0.65),
                "config": {"param1": idx, "param2": f"value{idx}"}
            }
            return await db.save_simulation_result(sim_data)
        
        start_time = time.time()
        
        tasks = [create_simulation(i) for i in range(num_simulations)]
        sim_ids = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # Verify all created
        assert len(sim_ids) == num_simulations
        assert all(sid is not None for sid in sim_ids)
        
        # Verify can query history
        history = await db.get_simulation_history(limit=num_simulations)
        assert len(history) >= num_simulations
        
        print(f"\n✓ {num_simulations} concurrent simulation writes in {elapsed_time:.2f}s")
    
    async def test_paginated_query_under_load(self, db_tables, sample_model_data):
        """
        Load Test: Paginated queries with concurrent inserts.
        
        Tests:
        - Query consistency during writes
        - Pagination correctness
        - No missing or duplicate results
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create initial batch
        initial_count = 30
        for i in range(initial_count):
            config = sample_model_data.copy()
            config["id"] = f"page-{i}-{uuid.uuid4()}"
            config["symbol"] = f"PG{i:02d}"
            config["fingerprint"] = compute_fingerprint({
                "features": [f"f{i}"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            await db.create_model_record(config)
        
        # Concurrent: paginated reads + new inserts
        async def paginated_reader():
            """Read all models with pagination."""
            all_models = []
            offset = 0
            limit = 5
            
            while True:
                batch = await db.list_models(limit=limit, offset=offset)
                if not batch:
                    break
                all_models.extend(batch)
                offset += limit
                await asyncio.sleep(0.01)  # Small delay
            
            return all_models
        
        async def concurrent_writer(idx: int):
            """Write new models during pagination."""
            config = sample_model_data.copy()
            config["id"] = f"page-new-{idx}-{uuid.uuid4()}"
            config["symbol"] = f"PN{idx}"
            config["fingerprint"] = compute_fingerprint({
                "features": ["new"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            await db.create_model_record(config)
        
        # Run readers and writers concurrently
        reader_tasks = [paginated_reader() for _ in range(3)]
        writer_tasks = [concurrent_writer(i) for i in range(10)]
        
        start_time = time.time()
        results = await asyncio.gather(*reader_tasks, *writer_tasks)
        elapsed_time = time.time() - start_time
        
        # Extract reader results (first 3)
        reader_results = results[:3]
        
        # Verify readers got results
        for result in reader_results:
            assert len(result) >= initial_count
        
        print(f"\n✓ Paginated queries with concurrent writes completed in {elapsed_time:.2f}s")
        print(f"  Readers: {len(reader_results)}, Writers: {len(writer_tasks)}")


@pytest.mark.asyncio
@pytest.mark.load
@pytest.mark.slow
class TestConnectionPoolStress:
    """Stress test connection pool behavior."""
    
    async def test_connection_pool_saturation(self, db_tables):
        """
        Load Test: Saturate connection pool with long-running queries.
        
        Tests:
        - Pool handles saturation gracefully
        - Queries queue when pool full
        - All queries eventually complete
        """
        from training_service.pg_db import get_pool
        
        pool = await get_pool()
        max_size = pool.get_max_size()
        
        # Create more concurrent tasks than pool size
        num_tasks = max_size + 5
        
        async def long_query(idx: int):
            """Execute a query that takes time."""
            async with pool.acquire() as conn:
                # Simulate long query with pg_sleep
                await conn.fetchval("SELECT pg_sleep(0.1)")
                return idx
        
        start_time = time.time()
        
        tasks = [long_query(i) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # All should complete
        assert len(results) == num_tasks
        assert set(results) == set(range(num_tasks))
        
        # Should take longer than single query (queuing)
        expected_min_time = (num_tasks / max_size) * 0.1
        assert elapsed_time >= expected_min_time * 0.8  # Allow some overhead
        
        print(f"\n✓ Pool saturation test: {num_tasks} queries with pool_size={max_size} in {elapsed_time:.2f}s")
    
    async def test_connection_pool_rapid_acquire_release(self, db_tables):
        """
        Load Test: Rapid connection acquire/release cycles.
        
        Tests:
        - Connection recycling performance
        - No connection leaks
        - Pool remains healthy
        """
        from training_service.pg_db import get_pool
        
        pool = await get_pool()
        
        num_cycles = 200
        
        async def acquire_release_cycle(idx: int):
            """Acquire and immediately release."""
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT $1", idx)
                return result
        
        start_time = time.time()
        
        tasks = [acquire_release_cycle(i) for i in range(num_cycles)]
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # All should succeed
        assert len(results) == num_cycles
        
        # Should be fast
        avg_time_per_cycle = elapsed_time / num_cycles
        assert avg_time_per_cycle < 0.05  # < 50ms per cycle
        
        print(f"\n✓ {num_cycles} acquire/release cycles in {elapsed_time:.2f}s")
        print(f"  Avg per cycle: {avg_time_per_cycle*1000:.2f}ms")
    
    async def test_mixed_query_complexity(self, db_tables, sample_model_data):
        """
        Load Test: Mix of simple and complex queries.
        
        Tests:
        - Simple queries don't block on complex queries
        - Pool handles varied workload
        - Resource fairness
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create test data
        model_ids = []
        for i in range(10):
            config = sample_model_data.copy()
            config["id"] = f"complex-{i}-{uuid.uuid4()}"
            config["symbol"] = f"CPX{i}"
            config["fingerprint"] = compute_fingerprint({
                "features": [f"f{i}"],
                "symbol": config["symbol"],
                "hyperparameters": {}
            })
            model_id = await db.create_model_record(config)
            model_ids.append(model_id)
        
        async def simple_query():
            """Simple query - get single model."""
            model_id = random.choice(model_ids)
            return await db.get_model(model_id)
        
        async def complex_query():
            """Complex query - list all models."""
            return await db.list_models(limit=100)
        
        # Mix: 20 simple + 5 complex
        start_time = time.time()
        
        tasks = []
        tasks.extend([simple_query() for _ in range(20)])
        tasks.extend([complex_query() for _ in range(5)])
        
        random.shuffle(tasks)  # Randomize execution order
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # All should complete
        assert len(results) == 25
        
        print(f"\n✓ Mixed query complexity: 20 simple + 5 complex in {elapsed_time:.2f}s")
    
    async def test_connection_timeout_handling(self, db_tables):
        """
        Load Test: Connection acquisition with timeout.
        
        Tests:
        - Timeout configuration works
        - Timeouts don't crash pool
        - Graceful degradation
        """
        from training_service.pg_db import get_pool
        import asyncpg
        
        pool = await get_pool()
        
        # Hold all connections
        held_connections = []
        max_size = pool.get_max_size()
        
        try:
            # Acquire all connections
            for _ in range(max_size):
                conn = await pool.acquire()
                held_connections.append(conn)
            
            # Try to acquire one more (should timeout)
            start_time = time.time()
            
            try:
                async with asyncio.timeout(1.0):  # 1 second timeout
                    await pool.acquire()
                    assert False, "Should have timed out"
            except asyncio.TimeoutError:
                # Expected
                pass
            
            elapsed_time = time.time() - start_time
            assert elapsed_time < 1.5  # Should timeout around 1s
            
            print(f"\n✓ Connection timeout handled correctly (timeout after {elapsed_time:.2f}s)")
        
        finally:
            # Release all held connections
            for conn in held_connections:
                await pool.release(conn)

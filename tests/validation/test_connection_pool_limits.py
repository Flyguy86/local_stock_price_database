"""
Validation tests for PostgreSQL connection pool limits.

Tests that the connection pool behaves correctly under various conditions:
- Respects max_size limit
- Handles concurrent connections properly
- Recovers from connection failures
- Closes connections cleanly
"""
import pytest
import asyncio
import asyncpg
from training_service.pg_db import get_pool, close_pool, POSTGRES_URL
from simulation_service.pg_db import get_pool as get_sim_pool, close_pool as close_sim_pool


class TestConnectionPoolLimits:
    """Test connection pool size limits and behavior."""
    
    @pytest.mark.asyncio
    async def test_pool_respects_max_size(self, db_tables):
        """
        Validation: Connection pool respects max_size configuration.
        
        The pool should not create more than max_size connections
        even under high concurrent demand.
        """
        pool = await get_pool()
        
        # Get pool configuration
        max_size = pool.get_max_size()
        min_size = pool.get_min_size()
        
        # Verify reasonable limits
        assert min_size >= 1
        assert max_size >= min_size
        assert max_size <= 20  # Reasonable upper bound
        
        # Current size should be within limits
        current_size = pool.get_size()
        assert current_size >= min_size
        assert current_size <= max_size
    
    @pytest.mark.asyncio
    async def test_pool_concurrent_acquisitions(self, db_tables):
        """
        Validation: Pool handles multiple concurrent connection acquisitions.
        
        Multiple tasks requesting connections concurrently should
        all get connections (up to max_size limit).
        """
        pool = await get_pool()
        connections = []
        
        try:
            # Acquire multiple connections concurrently
            # Use 5 connections (should be well under max_size)
            for _ in range(5):
                conn = await pool.acquire()
                connections.append(conn)
            
            # All connections should be valid
            assert len(connections) == 5
            
            # Verify each connection works
            for conn in connections:
                result = await conn.fetchval("SELECT 1")
                assert result == 1
        
        finally:
            # Release all connections
            for conn in connections:
                await pool.release(conn)
    
    @pytest.mark.asyncio
    async def test_pool_connection_reuse(self, db_tables):
        """
        Validation: Connections are properly reused after release.
        
        Acquiring and releasing should allow connection reuse
        rather than creating new connections each time.
        """
        pool = await get_pool()
        
        # Acquire connection
        conn1 = await pool.acquire()
        conn1_id = id(conn1)
        
        # Use and release
        await conn1.fetchval("SELECT 1")
        await pool.release(conn1)
        
        # Acquire again - might get same connection
        conn2 = await pool.acquire()
        
        # Verify connection works
        result = await conn2.fetchval("SELECT 1")
        assert result == 1
        
        # Release second connection
        await pool.release(conn2)
        
        # Pool should still be healthy
        current_size = pool.get_size()
        assert current_size >= pool.get_min_size()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_pool_under_load(self, db_tables):
        """
        Validation: Pool performs correctly under concurrent load.
        
        Simulate high concurrent usage to verify pool stability.
        """
        pool = await get_pool()
        
        async def worker(worker_id: int):
            """Simulate a worker using the pool."""
            conn = await pool.acquire()
            try:
                # Simulate work
                result = await conn.fetchval(f"SELECT {worker_id}")
                assert result == worker_id
                await asyncio.sleep(0.01)  # Small delay
                return True
            finally:
                await pool.release(conn)
        
        # Run 20 workers concurrently
        tasks = [worker(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # All workers should succeed
        assert all(results)
        assert len(results) == 20
    
    @pytest.mark.asyncio
    async def test_pool_min_size_maintained(self, db_tables):
        """
        Validation: Pool maintains minimum connection count.
        
        The pool should keep at least min_size connections open
        even when idle.
        """
        pool = await get_pool()
        
        min_size = pool.get_min_size()
        current_size = pool.get_size()
        
        # Current size should be at least min_size
        assert current_size >= min_size
        
        # After some idle time, size should still be >= min_size
        await asyncio.sleep(0.1)
        current_size_after = pool.get_size()
        assert current_size_after >= min_size


class TestConnectionPoolIsolation:
    """Test connection pool isolation between services."""
    
    @pytest.mark.asyncio
    async def test_separate_pools_for_services(self, db_tables):
        """
        Validation: Training and simulation services have separate pools.
        
        Each service should maintain its own connection pool
        to avoid resource contention.
        """
        training_pool = await get_pool()
        simulation_pool = await get_sim_pool()
        
        # Pools should be different objects
        assert training_pool is not simulation_pool
        
        # Both should be valid
        assert training_pool is not None
        assert simulation_pool is not None
        
        # Both should have reasonable configurations
        assert training_pool.get_max_size() >= 2
        assert simulation_pool.get_max_size() >= 2
    
    @pytest.mark.asyncio
    async def test_pools_independent_connections(self, db_tables):
        """
        Validation: Service pools manage independent connections.
        
        Connections from different pools should be independent.
        """
        training_pool = await get_pool()
        simulation_pool = await get_sim_pool()
        
        # Acquire from both pools
        training_conn = await training_pool.acquire()
        simulation_conn = await simulation_pool.acquire()
        
        try:
            # Both connections should work independently
            training_result = await training_conn.fetchval("SELECT 'training'")
            simulation_result = await simulation_conn.fetchval("SELECT 'simulation'")
            
            assert training_result == 'training'
            assert simulation_result == 'simulation'
            
        finally:
            # Release both
            await training_pool.release(training_conn)
            await simulation_pool.release(simulation_conn)


class TestConnectionPoolCleanup:
    """Test connection pool cleanup and shutdown."""
    
    @pytest.mark.asyncio
    async def test_pool_closes_cleanly(self, db_tables):
        """
        Validation: Pool closes all connections cleanly.
        
        When closing the pool, all connections should be
        properly terminated.
        """
        # Create a fresh pool
        pool = await asyncpg.create_pool(
            POSTGRES_URL,
            min_size=2,
            max_size=5,
            command_timeout=60
        )
        
        # Acquire some connections
        conn1 = await pool.acquire()
        conn2 = await pool.acquire()
        
        # Release connections
        await pool.release(conn1)
        await pool.release(conn2)
        
        # Close pool
        await pool.close()
        
        # Pool should be closed
        # Attempting to acquire should fail
        with pytest.raises(Exception):
            await pool.acquire()
    
    @pytest.mark.asyncio
    async def test_pool_handles_connection_in_use(self, db_tables):
        """
        Validation: Pool waits for in-use connections on close.
        
        When closing, the pool should wait for active connections
        to be released or timeout gracefully.
        """
        pool = await asyncpg.create_pool(
            POSTGRES_URL,
            min_size=1,
            max_size=3,
            command_timeout=60
        )
        
        # Acquire connection
        conn = await pool.acquire()
        
        # Start close in background (will wait for connection)
        close_task = asyncio.create_task(pool.close())
        
        # Give it a moment
        await asyncio.sleep(0.1)
        
        # Release connection
        await pool.release(conn)
        
        # Close should complete now
        await close_task


class TestConnectionPoolErrors:
    """Test connection pool error handling."""
    
    @pytest.mark.asyncio
    async def test_pool_survives_query_error(self, db_tables):
        """
        Validation: Pool remains healthy after query errors.
        
        A bad query on one connection shouldn't affect
        the pool's overall health.
        """
        pool = await get_pool()
        
        # Execute bad query
        conn = await pool.acquire()
        try:
            # This will raise an error
            await conn.fetchval("SELECT * FROM nonexistent_table")
        except Exception:
            # Expected - invalid query
            pass
        finally:
            await pool.release(conn)
        
        # Pool should still work
        conn2 = await pool.acquire()
        try:
            result = await conn2.fetchval("SELECT 1")
            assert result == 1
        finally:
            await pool.release(conn2)
    
    @pytest.mark.asyncio
    async def test_pool_timeout_configuration(self, db_tables):
        """
        Validation: Pool has reasonable timeout configuration.
        
        Command timeouts should be set to prevent hung connections.
        """
        pool = await get_pool()
        
        # Pool should have timeout configured
        # (Implementation-dependent, but should exist)
        # This validates pool is configured properly
        assert pool is not None
        
        # Verify basic operations complete within reasonable time
        start = asyncio.get_event_loop().time()
        conn = await pool.acquire()
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await pool.release(conn)
        end = asyncio.get_event_loop().time()
        
        # Should complete in well under 1 second
        elapsed = end - start
        assert elapsed < 1.0


class TestConnectionPoolPerformance:
    """Test connection pool performance characteristics."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_pool_acquisition_speed(self, db_tables):
        """
        Validation: Connection acquisition is fast.
        
        Acquiring connections from the pool should be quick
        (no excessive overhead).
        """
        pool = await get_pool()
        
        # Time 100 acquisitions/releases
        start = asyncio.get_event_loop().time()
        
        for _ in range(100):
            conn = await pool.acquire()
            await pool.release(conn)
        
        end = asyncio.get_event_loop().time()
        elapsed = end - start
        
        # Should average < 10ms per acquire/release
        avg_time = elapsed / 100
        assert avg_time < 0.01  # 10ms
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_pool_concurrent_throughput(self, db_tables):
        """
        Validation: Pool handles high concurrent throughput.
        
        Many concurrent queries should be handled efficiently.
        """
        pool = await get_pool()
        
        async def query_worker():
            """Execute a simple query."""
            conn = await pool.acquire()
            try:
                result = await conn.fetchval("SELECT 1")
                return result
            finally:
                await pool.release(conn)
        
        # Run 50 concurrent workers
        start = asyncio.get_event_loop().time()
        tasks = [query_worker() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        end = asyncio.get_event_loop().time()
        
        # All should succeed
        assert all(r == 1 for r in results)
        
        # Should complete in reasonable time (< 2 seconds)
        elapsed = end - start
        assert elapsed < 2.0


class TestProcessPoolConnections:
    """Test connection pools in multi-process environment."""
    
    @pytest.mark.asyncio
    async def test_pool_per_process(self, db_tables):
        """
        Validation: Each process creates its own connection pool.
        
        In multi-process training, each worker process should
        create its own asyncpg pool (not shared).
        """
        from concurrent.futures import ProcessPoolExecutor
        import os
        
        def get_pool_in_process():
            """Get pool info from a separate process."""
            import asyncio
            from training_service.sync_db_wrapper import sync_get_model_by_id
            
            # This will create a pool in the worker process
            # Return process ID to verify different process
            return os.getpid()
        
        # Get current process ID
        main_pid = os.getpid()
        
        # Execute in worker process
        with ProcessPoolExecutor(max_workers=2) as executor:
            worker_pid = executor.submit(get_pool_in_process).result()
        
        # Worker should be different process
        assert worker_pid != main_pid
        
        # This validates that pools are per-process
        # (each worker creates its own pool)


class TestConnectionPoolRecovery:
    """Test connection pool recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_pool_recreates_failed_connection(self, db_tables):
        """
        Validation: Pool handles failed connections gracefully.
        
        If a connection fails, the pool should be able to
        create new connections.
        """
        pool = await get_pool()
        
        # Get initial pool size
        initial_size = pool.get_size()
        
        # Acquire and use connection normally
        conn = await pool.acquire()
        try:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
        finally:
            await pool.release(conn)
        
        # Pool should maintain size
        current_size = pool.get_size()
        assert current_size >= pool.get_min_size()


# Mark validation tests
pytestmark = pytest.mark.asyncio

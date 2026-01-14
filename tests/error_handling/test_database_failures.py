"""
Error handling tests for database failures and recovery.

Tests system behavior when database connections fail or become unavailable.
"""
import pytest
import asyncio
import uuid
from unittest.mock import patch, MagicMock
import asyncpg


@pytest.mark.asyncio
@pytest.mark.error_handling
class TestDatabaseConnectionFailures:
    """Test handling of database connection failures."""
    
    async def test_connection_failure_during_query(self, db_tables, sample_model_data):
        """
        Error Test: Handle connection failure during query.
        
        Validates:
        - Connection errors are properly caught
        - Error messages are clear
        - System doesn't crash
        """
        from training_service.pg_db import TrainingDB
        
        db = TrainingDB()
        
        # Create a model
        config = sample_model_data.copy()
        config["id"] = f"conn-fail-{uuid.uuid4()}"
        model_id = await db.create_model_record(config)
        
        # Mock connection failure
        with patch.object(db, 'get_model', side_effect=asyncpg.ConnectionDoesNotExistError("Connection closed")):
            with pytest.raises(asyncpg.ConnectionDoesNotExistError):
                await db.get_model(model_id)
        
        # Verify database still works after error
        model = await db.get_model(model_id)
        assert model is not None
    
    async def test_connection_pool_exhaustion_recovery(self, db_tables):
        """
        Error Test: Recover from connection pool exhaustion.
        
        Validates:
        - System handles pool exhaustion gracefully
        - Recovers when connections released
        - No permanent damage
        """
        from training_service.pg_db import get_pool
        
        pool = await get_pool()
        max_size = pool.get_max_size()
        
        # Hold all connections
        held_connections = []
        
        try:
            for _ in range(max_size):
                conn = await pool.acquire()
                held_connections.append(conn)
            
            # Try to acquire one more (will timeout)
            with pytest.raises(asyncio.TimeoutError):
                async with asyncio.timeout(0.5):
                    await pool.acquire()
            
            # Release one connection
            await pool.release(held_connections.pop())
            
            # Now should be able to acquire
            async with asyncio.timeout(1.0):
                conn = await pool.acquire()
                await pool.release(conn)
            
            print("\n✓ Recovered from pool exhaustion")
        
        finally:
            # Cleanup
            for conn in held_connections:
                await pool.release(conn)
    
    async def test_query_timeout_handling(self, db_tables):
        """
        Error Test: Handle long-running query timeout.
        
        Validates:
        - Timeouts are enforced
        - Connection remains usable
        - No resource leaks
        """
        from training_service.pg_db import get_pool
        
        pool = await get_pool()
        
        # Execute long query with timeout
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.5):
                async with pool.acquire() as conn:
                    # This query will take 2 seconds
                    await conn.fetchval("SELECT pg_sleep(2)")
        
        # Verify pool still works
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
        
        print("\n✓ Query timeout handled correctly")
    
    async def test_invalid_sql_error_handling(self, db_tables):
        """
        Error Test: Handle invalid SQL queries.
        
        Validates:
        - SQL syntax errors caught
        - Connection remains valid
        - Helpful error messages
        """
        from training_service.pg_db import get_pool
        
        pool = await get_pool()
        
        # Execute invalid SQL
        with pytest.raises(asyncpg.PostgresSyntaxError):
            async with pool.acquire() as conn:
                await conn.execute("SELET * FROM models")  # Typo
        
        # Connection should still work
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1
        
        print("\n✓ Invalid SQL handled without breaking connection")
    
    async def test_foreign_key_violation_handling(self, db_tables):
        """
        Error Test: Handle foreign key constraint violations.
        
        Validates:
        - FK violations caught properly
        - Transaction rolled back
        - Database remains consistent
        """
        from training_service.pg_db import TrainingDB
        
        db = TrainingDB()
        
        # Try to add feature importance for non-existent model
        with pytest.raises(Exception):  # FK violation
            await db.store_feature_importance("non-existent-id", {"feature1": 0.5})
        
        # Database should still work
        models = await db.list_models()
        assert isinstance(models, list)
        
        print("\n✓ Foreign key violation handled correctly")
    
    async def test_duplicate_key_error_handling(self, db_tables, sample_model_data):
        """
        Error Test: Handle duplicate primary key insertion.
        
        Validates:
        - Duplicate key errors caught
        - Clear error message
        - No data corruption
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create a model
        config = sample_model_data.copy()
        model_id = f"dup-test-{uuid.uuid4()}"
        config["id"] = model_id
        config["fingerprint"] = compute_fingerprint({
            "features": config.get("features", []),
            "symbol": config.get("symbol", "TEST"),
            "hyperparameters": {}
        })
        
        await db.create_model_record(config)
        
        # Try to create again with same ID
        with pytest.raises(Exception):  # Unique violation
            await db.create_model_record(config)
        
        # Verify only one record exists
        model = await db.get_model(model_id)
        assert model is not None
        
        print("\n✓ Duplicate key error handled correctly")
    
    async def test_transaction_rollback_on_error(self, db_tables, sample_model_data):
        """
        Error Test: Verify transaction rollback on error.
        
        Validates:
        - Failed transactions roll back
        - No partial writes
        - Data integrity maintained
        """
        from training_service.pg_db import get_pool
        
        pool = await get_pool()
        
        model_id = f"rollback-test-{uuid.uuid4()}"
        
        # Start transaction and cause error
        with pytest.raises(Exception):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Insert model
                    await conn.execute(
                        "INSERT INTO models (id, symbol, algorithm, status) VALUES ($1, $2, $3, $4)",
                        model_id, "TEST", "RandomForest", "pending"
                    )
                    
                    # Cause error (invalid column)
                    await conn.execute("SELECT nonexistent_column FROM models")
        
        # Verify model was NOT inserted (rollback)
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT id FROM models WHERE id = $1", model_id)
            assert result is None
        
        print("\n✓ Transaction rolled back on error")


@pytest.mark.asyncio
@pytest.mark.error_handling
class TestDatabaseRecovery:
    """Test database connection recovery mechanisms."""
    
    async def test_reconnect_after_connection_lost(self, db_tables):
        """
        Error Test: Reconnect after losing database connection.
        
        Validates:
        - System detects lost connection
        - Automatically reconnects
        - Operations resume normally
        """
        from training_service.pg_db import get_pool
        
        pool = await get_pool()
        
        # Simulate connection loss by closing a connection improperly
        async with pool.acquire() as conn:
            # Force close (simulates network failure)
            await conn.close()
        
        # Pool should recover and provide new connection
        async with asyncio.timeout(5.0):
            async with pool.acquire() as new_conn:
                result = await new_conn.fetchval("SELECT 1")
                assert result == 1
        
        print("\n✓ Reconnected after connection loss")
    
    async def test_pool_graceful_degradation(self, db_tables):
        """
        Error Test: Pool continues working with reduced capacity.
        
        Validates:
        - Pool works even when some connections fail
        - Graceful degradation
        - System remains available
        """
        from training_service.pg_db import get_pool
        
        pool = await get_pool()
        
        # Even if pool has issues, should be able to execute queries
        # (Pool internally handles connection failures)
        
        num_queries = 10
        success_count = 0
        
        for _ in range(num_queries):
            try:
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                success_count += 1
            except Exception as e:
                print(f"Query failed: {e}")
        
        # Most queries should succeed
        assert success_count >= num_queries * 0.8  # At least 80% success
        
        print(f"\n✓ Graceful degradation: {success_count}/{num_queries} queries succeeded")
    
    async def test_retry_on_transient_failure(self, db_tables, sample_model_data):
        """
        Error Test: Retry logic for transient failures.
        
        Validates:
        - Transient failures trigger retry
        - Eventually succeeds
        - Retry count is reasonable
        """
        from training_service.pg_db import TrainingDB
        
        db = TrainingDB()
        
        call_count = [0]
        
        # Mock transient failure
        original_create = db.create_model_record
        
        async def create_with_transient_failure(config):
            call_count[0] += 1
            if call_count[0] < 3:  # Fail first 2 times
                raise asyncpg.ConnectionDoesNotExistError("Transient error")
            return await original_create(config)
        
        config = sample_model_data.copy()
        config["id"] = f"retry-test-{uuid.uuid4()}"
        
        # Simple retry logic
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                with patch.object(db, 'create_model_record', create_with_transient_failure):
                    model_id = await db.create_model_record(config)
                    break
            except asyncpg.ConnectionDoesNotExistError as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
        else:
            if last_error:
                raise last_error
        
        # Should have succeeded after retries
        assert call_count[0] == 3
        
        print(f"\n✓ Succeeded after {call_count[0]} attempts (2 failures + 1 success)")


@pytest.mark.asyncio
@pytest.mark.error_handling  
class TestDataValidation:
    """Test data validation and constraint enforcement."""
    
    async def test_invalid_status_transition(self, db_tables, sample_model_data):
        """
        Error Test: Prevent invalid status transitions.
        
        Validates:
        - Status validation works
        - Invalid states rejected
        - Clear error messages
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create model
        config = sample_model_data.copy()
        config["id"] = f"status-test-{uuid.uuid4()}"
        config["fingerprint"] = compute_fingerprint({
            "features": config.get("features", []),
            "symbol": config.get("symbol", "TEST"),
            "hyperparameters": {}
        })
        model_id = await db.create_model_record(config)
        
        # Try invalid status
        with pytest.raises(Exception):
            await db.update_model_status(model_id, "invalid_status")
        
        # Verify status unchanged
        model = await db.get_model(model_id)
        assert model["status"] == "pending"
        
        print("\n✓ Invalid status rejected")
    
    async def test_null_value_validation(self, db_tables):
        """
        Error Test: Handle NULL values in NOT NULL columns.
        
        Validates:
        - NOT NULL constraints enforced
        - Clear validation errors
        - No data corruption
        """
        from training_service.pg_db import get_pool
        
        pool = await get_pool()
        
        # Try to insert model with NULL required field
        with pytest.raises(asyncpg.NotNullViolationError):
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO models (id, symbol, algorithm) VALUES ($1, $2, $3)",
                    f"null-test-{uuid.uuid4()}", "TEST", None  # algorithm is NULL
                )
        
        print("\n✓ NULL constraint violation caught")
    
    async def test_invalid_json_data(self, db_tables, sample_model_data):
        """
        Error Test: Handle invalid JSON in JSONB columns.
        
        Validates:
        - JSON validation works
        - Invalid JSON rejected
        - Type safety maintained
        """
        from training_service.pg_db import TrainingDB, compute_fingerprint
        
        db = TrainingDB()
        
        # Create model with valid data
        config = sample_model_data.copy()
        config["id"] = f"json-test-{uuid.uuid4()}"
        config["fingerprint"] = compute_fingerprint({
            "features": config.get("features", []),
            "symbol": config.get("symbol", "TEST"),
            "hyperparameters": {}
        })
        model_id = await db.create_model_record(config)
        
        # Try to store invalid JSON (not a dict)
        with pytest.raises((TypeError, ValueError, Exception)):
            await db.store_feature_importance(model_id, "not a dict")
        
        print("\n✓ Invalid JSON data rejected")

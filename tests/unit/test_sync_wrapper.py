"""
Unit tests for training_service/sync_db_wrapper.py
Tests synchronous wrapper for async PostgreSQL operations.
"""
import pytest
import multiprocessing as mp
import uuid
import json
import time
from concurrent.futures import ProcessPoolExecutor


def test_basic_import():
    """Test that sync_db_wrapper can be imported."""
    from training_service.sync_db_wrapper import SyncDBWrapper
    
    assert SyncDBWrapper is not None


def create_model_in_process(model_data):
    """
    Helper function to create a model in a separate process.
    Tests that each process gets its own connection pool.
    """
    import os
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Set test database URL
    os.environ['POSTGRES_URL'] = os.environ.get(
        'TEST_POSTGRES_URL',
        'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
    )
    
    from training_service.sync_db_wrapper import db
    
    # Get current process ID
    pid = os.getpid()
    
    try:
        # Create model using sync wrapper
        db.create_model_record(model_data)
        
        # Update status
        db.update_model_status(
            model_data['id'],
            status='completed',
            metrics={'test': True}
        )
        
        # Get model back
        model = db.get_model(model_data['id'])
        
        return {
            'success': True,
            'pid': pid,
            'model_id': model_data['id'],
            'model_status': model.get('status') if model else None
        }
    except Exception as e:
        return {
            'success': False,
            'pid': pid,
            'error': str(e)
        }


def update_model_in_process(model_id, status):
    """Helper function to update a model from a separate process."""
    import os
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    os.environ['POSTGRES_URL'] = os.environ.get(
        'TEST_POSTGRES_URL',
        'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
    )
    
    from training_service.sync_db_wrapper import db
    
    pid = os.getpid()
    
    try:
        db.update_model_status(model_id, status=status)
        model = db.get_model(model_id)
        
        return {
            'success': True,
            'pid': pid,
            'status': model.get('status') if model else None
        }
    except Exception as e:
        return {
            'success': False,
            'pid': pid,
            'error': str(e)
        }


class TestSyncDBWrapper:
    """Test suite for synchronous database wrapper."""
    
    def test_wrapper_initialization(self):
        """Test that wrapper can be initialized."""
        from training_service.sync_db_wrapper import SyncDBWrapper
        
        wrapper = SyncDBWrapper()
        assert wrapper is not None
    
    def test_wrapper_has_methods(self):
        """Test that wrapper has all required methods used by trainer.py."""
        from training_service.sync_db_wrapper import db
        
        # Only check for methods actually used in trainer.py
        required_methods = [
            'create_model_record',
            'update_model_status',
            'get_model',
        ]
        
        for method in required_methods:
            assert hasattr(db, method), f"Wrapper should have {method} method"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sync_wrapper_basic_operations(self, db_tables):
        """Test basic CRUD operations through sync wrapper."""
        import os
        from training_service.sync_db_wrapper import SyncDBWrapper
        import training_service.pg_db as pg_db
        
        # Set test URL
        original_url = pg_db.POSTGRES_URL
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
        )
        pg_db.POSTGRES_URL = test_url
        
        try:
            wrapper = SyncDBWrapper()
            
            # Create model
            model_id = str(uuid.uuid4())
            model_data = {
                'id': model_id,
                'algorithm': 'RandomForest',
                'symbol': 'RDDT',
                'target_col': 'close',
                'feature_cols': json.dumps(['sma_20']),
                'hyperparameters': json.dumps({'n_estimators': 100}),
                'status': 'preprocessing',
                'timeframe': '1m'
            }
            
            wrapper.create_model_record(model_data)
            
            # Get model
            model = wrapper.get_model(model_id)
            assert model is not None
            assert model['id'] == model_id
            
            # Update status
            wrapper.update_model_status(model_id, status='completed')
            
            model = wrapper.get_model(model_id)
            assert model['status'] == 'completed'
            
        finally:
            pg_db.POSTGRES_URL = original_url
    
    @pytest.mark.integration
    def test_multiple_processes_create_models(self):
        """Test that multiple processes can create models independently."""
        import os
        
        # Ensure test database URL is set
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
        )
        os.environ['TEST_POSTGRES_URL'] = test_url
        
        # Create model data for 3 processes
        model_data_list = []
        for i in range(3):
            model_data_list.append({
                'id': str(uuid.uuid4()),
                'algorithm': 'RandomForest',
                'symbol': f'SYM{i}',
                'target_col': 'close',
                'feature_cols': json.dumps(['sma_20']),
                'hyperparameters': json.dumps({'n_estimators': 100}),
                'status': 'preprocessing',
                'timeframe': '1m'
            })
        
        # Run in separate processes
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(create_model_in_process, data)
                for data in model_data_list
            ]
            
            results = [f.result() for f in futures]
        
        # Verify all succeeded
        for result in results:
            assert result['success'] is True, f"Process {result['pid']} failed: {result.get('error')}"
            assert result['model_status'] == 'completed'
        
        # Verify different PIDs (processes)
        pids = [r['pid'] for r in results]
        assert len(set(pids)) > 1, "Should have used multiple processes"
    
    @pytest.mark.integration
    def test_concurrent_model_updates(self):
        """Test concurrent updates from multiple processes."""
        import os
        
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
        )
        os.environ['TEST_POSTGRES_URL'] = test_url
        
        # Create a model first
        model_id = str(uuid.uuid4())
        model_data = {
            'id': model_id,
            'algorithm': 'RandomForest',
            'symbol': 'RDDT',
            'target_col': 'close',
            'feature_cols': json.dumps(['sma_20']),
            'hyperparameters': json.dumps({'n_estimators': 100}),
            'status': 'preprocessing',
            'timeframe': '1m'
        }
        
        # Create in main process
        from training_service.sync_db_wrapper import db
        import training_service.pg_db as pg_db
        original_url = pg_db.POSTGRES_URL
        pg_db.POSTGRES_URL = test_url
        
        try:
            db.create_model_record(model_data)
            
            # Update from multiple processes
            statuses = ['training', 'validating', 'completed']
            
            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(update_model_in_process, model_id, status)
                    for status in statuses
                ]
                
                results = [f.result() for f in futures]
            
            # All should succeed
            for result in results:
                assert result['success'] is True, f"Update failed: {result.get('error')}"
            
            # Final status should be one of the attempted statuses
            final_model = db.get_model(model_id)
            assert final_model['status'] in statuses
            
        finally:
            pg_db.POSTGRES_URL = original_url
    
    @pytest.mark.integration
    def test_process_isolation(self):
        """Test that each process has isolated connection pool."""
        import os
        
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
        )
        os.environ['TEST_POSTGRES_URL'] = test_url
        
        # Create models in parallel - each should get own pool
        model_data_list = []
        for i in range(4):
            model_data_list.append({
                'id': str(uuid.uuid4()),
                'algorithm': 'RandomForest',
                'symbol': f'TEST{i}',
                'target_col': 'close',
                'feature_cols': json.dumps(['sma_20']),
                'hyperparameters': json.dumps({'n_estimators': 10}),
                'status': 'preprocessing',
                'timeframe': '1m'
            })
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(create_model_in_process, data)
                for data in model_data_list
            ]
            
            results = [f.result() for f in futures]
        
        # All should succeed
        success_count = sum(1 for r in results if r['success'])
        assert success_count >= 3, f"At least 3 processes should succeed, got {success_count}"
        
        # Should have multiple different PIDs
        pids = [r['pid'] for r in results if r['success']]
        unique_pids = len(set(pids))
        assert unique_pids >= 2, f"Should use multiple processes, got {unique_pids} unique PIDs"


@pytest.mark.slow
class TestSyncWrapperPerformance:
    """Performance tests for sync wrapper (slow, optional)."""
    
    def test_many_sequential_operations(self):
        """Test many sequential operations don't leak connections."""
        import os
        from training_service.sync_db_wrapper import db
        import training_service.pg_db as pg_db
        
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@postgres:5432/strategy_factory_test'
        )
        
        original_url = pg_db.POSTGRES_URL
        pg_db.POSTGRES_URL = test_url
        
        try:
            # Create and query many models
            for i in range(20):
                model_id = str(uuid.uuid4())
                model_data = {
                    'id': model_id,
                    'algorithm': 'RandomForest',
                    'symbol': 'RDDT',
                    'target_col': 'close',
                    'feature_cols': json.dumps(['sma_20']),
                    'hyperparameters': json.dumps({'n_estimators': 10}),
                    'status': 'preprocessing',
                    'timeframe': '1m'
                }
                
                db.create_model_record(model_data)
                model = db.get_model(model_id)
                assert model is not None
            
            # Should still be able to list models
            models = db.list_models(limit=25)
            assert len(models) >= 20
            
        finally:
            pg_db.POSTGRES_URL = original_url


@pytest.mark.integration
class TestEventLoopHandling:
    """
    Regression tests for event loop conflicts (Issue: RuntimeError 'This event loop is already running').
    
    When SyncDBWrapper is used from worker processes (ProcessPoolExecutor), there may
    already be a running event loop. The wrapper must handle this gracefully.
    """
    
    def test_execute_async_with_no_event_loop(self):
        """Test that _execute_async works when no event loop exists."""
        from training_service.sync_db_wrapper import SyncDBWrapper
        import asyncio
        
        # Ensure no event loop is set
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except RuntimeError:
            pass
        
        asyncio.set_event_loop(None)
        
        wrapper = SyncDBWrapper()
        
        # Should create a new event loop and execute
        async def simple_coro():
            await asyncio.sleep(0.001)
            return "success"
        
        result = wrapper._execute_async(simple_coro())
        assert result == "success"
    
    def test_execute_async_with_running_event_loop(self):
        """
        Test that _execute_async works when an event loop is already running.
        
        This is the regression test for: RuntimeError: This event loop is already running
        """
        from training_service.sync_db_wrapper import SyncDBWrapper
        import asyncio
        
        wrapper = SyncDBWrapper()
        
        async def outer_coro():
            """Simulates the context where an event loop is already running."""
            # Inside this coroutine, there's a running event loop
            
            # Create an async operation to execute
            async def inner_coro():
                await asyncio.sleep(0.001)
                return "executed_with_running_loop"
            
            # This should NOT raise RuntimeError
            result = wrapper._execute_async(inner_coro())
            return result
        
        # Run in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(outer_coro())
            assert result == "executed_with_running_loop"
        finally:
            loop.close()
    
    def test_database_operations_from_running_loop(self, training_db_fixture):
        """
        Test database operations when called from within a running event loop.
        
        This simulates the exact scenario that caused the bug: worker process
        has an event loop, and tries to call sync wrapper methods.
        """
        from training_service.sync_db_wrapper import SyncDBWrapper
        import asyncio
        
        wrapper = SyncDBWrapper()
        model_id = str(uuid.uuid4())
        
        async def worker_simulation():
            """Simulates a worker process with an active event loop."""
            # Create model data
            model_data = {
                'id': model_id,
                'name': 'test-event-loop',
                'algorithm': 'RandomForest',
                'symbol': 'TEST',
                'target_col': 'close',
                'feature_cols': json.dumps(['feature_1']),
                'hyperparameters': json.dumps({'n_estimators': 100}),
                'metrics': json.dumps({}),
                'status': 'pending',
                'timeframe': '1m',
                'fingerprint': 'test_loop_fp'
            }
            
            # These should NOT raise RuntimeError
            wrapper.create_model_record(model_data)
            
            # Update status
            wrapper.update_model_status(
                model_id,
                status='training',
                metrics=json.dumps({'progress': 0.5})
            )
            
            # Get model
            model = wrapper.get_model(model_id)
            assert model is not None
            assert model['status'] == 'training'
            
            return "success"
        
        # Run in event loop to simulate worker context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(worker_simulation())
            assert result == "success"
        finally:
            loop.close()
    
    def test_concurrent_database_ops_from_process_pool(self, training_db_fixture):
        """
        Test that multiple worker processes can use SyncDBWrapper concurrently.
        
        This is the real-world scenario: ProcessPoolExecutor workers all trying
        to use the sync wrapper simultaneously.
        """
        import os
        
        def worker_task(task_id):
            """Worker function that runs in a separate process."""
            from training_service.sync_db_wrapper import SyncDBWrapper
            import json
            import uuid
            
            # Each worker gets its own wrapper instance
            wrapper = SyncDBWrapper()
            
            model_id = str(uuid.uuid4())
            model_data = {
                'id': model_id,
                'name': f'concurrent-test-{task_id}',
                'algorithm': 'ElasticNet',
                'symbol': f'SYM{task_id}',
                'target_col': 'close',
                'feature_cols': json.dumps(['rsi', 'macd']),
                'hyperparameters': json.dumps({'alpha': 0.1}),
                'metrics': json.dumps({}),
                'status': 'pending',
                'timeframe': '5m',
                'fingerprint': f'concurrent_fp_{task_id}'
            }
            
            try:
                # Create model
                wrapper.create_model_record(model_data)
                
                # Update status
                wrapper.update_model_status(
                    model_id,
                    status='completed',
                    metrics=json.dumps({'accuracy': 0.85})
                )
                
                # Verify
                model = wrapper.get_model(model_id)
                assert model is not None
                assert model['status'] == 'completed'
                
                return {'success': True, 'model_id': model_id, 'task_id': task_id}
            except Exception as e:
                return {'success': False, 'error': str(e), 'task_id': task_id}
        
        # Run multiple workers concurrently
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_task, i) for i in range(8)]
            results = [f.result(timeout=30) for f in futures]
        
        # All should succeed
        assert all(r['success'] for r in results), f"Some workers failed: {results}"
        assert len(results) == 8
    
    def test_pool_creation_thread_safety(self):
        """
        Test that pool creation is thread-safe when called concurrently.
        
        Multiple threads trying to create the pool at the same time should
        result in only one pool being created (double-check locking).
        """
        from training_service.sync_db_wrapper import SyncDBWrapper
        import threading
        
        wrapper = SyncDBWrapper()
        pools_created = []
        
        async def create_pool_task():
            """Async task that triggers pool creation."""
            pool = await wrapper._get_or_create_pool()
            pools_created.append(id(pool))
            return pool
        
        def thread_worker():
            """Thread that executes the async task."""
            import asyncio
            result = wrapper._execute_async(create_pool_task())
            return result
        
        # Create multiple threads that all try to get the pool
        threads = [threading.Thread(target=thread_worker) for _ in range(10)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=10)
        
        # All threads should have gotten the SAME pool instance
        # (all pool IDs should be identical)
        assert len(set(pools_created)) == 1, "Multiple pools were created instead of one"
        assert len(pools_created) == 10, "Not all threads got a pool"
    
    def test_update_model_status_with_event_loop(self, training_db_fixture):
        """
        Specific regression test for the exact error from the logs.
        
        Error was: RuntimeError: This event loop is already running
        Location: training_service/sync_db_wrapper.py:128 in update_model_status
        """
        from training_service.sync_db_wrapper import SyncDBWrapper
        import asyncio
        
        wrapper = SyncDBWrapper()
        model_id = str(uuid.uuid4())
        
        # First create a model
        model_data = {
            'id': model_id,
            'algorithm': 'Ridge',
            'symbol': 'MSFT',
            'target_col': 'close',
            'feature_cols': json.dumps(['sma_50', 'rsi_14']),
            'hyperparameters': json.dumps({'alpha': 10.0}),
            'status': 'preprocessing',
            'timeframe': '1m',
            'fingerprint': 'regression_test_fp'
        }
        
        async def simulate_trainer_context():
            """
            Simulates the exact context where the bug occurred:
            trainer.py calling update_model_status from a worker process.
            """
            # Create model
            wrapper.create_model_record(model_data)
            
            # This is the exact call that failed in the logs
            # Should NOT raise RuntimeError
            wrapper.update_model_status(
                model_id,
                status='training',
                metrics=json.dumps({'loss': 0.5})
            )
            
            # Update again with error (as in failure scenario)
            wrapper.update_model_status(
                model_id,
                status='failed',
                error='Test error message'
            )
            
            # Verify final state
            model = wrapper.get_model(model_id)
            assert model['status'] == 'failed'
            assert model['error_message'] == 'Test error message'
            
            return True
        
        # Run with an active event loop (worker process context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(simulate_trainer_context())
            assert result is True
        finally:
            loop.close()
    
    def test_cleanup_without_event_loop_error(self):
        """
        Test that wrapper cleanup doesn't raise event loop errors.
        """
        from training_service.sync_db_wrapper import SyncDBWrapper
        
        wrapper = SyncDBWrapper()
        
        # Use the wrapper
        async def dummy_op():
            pool = await wrapper._get_or_create_pool()
            return pool
        
        wrapper._execute_async(dummy_op())
        
        # Cleanup should not raise
        try:
            wrapper.close()
        except RuntimeError as e:
            pytest.fail(f"Cleanup raised RuntimeError: {e}")


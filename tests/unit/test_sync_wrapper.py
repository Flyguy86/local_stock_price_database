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
        'postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test'
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
        'postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test'
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
        """Test that wrapper has all required methods."""
        from training_service.sync_db_wrapper import db
        
        required_methods = [
            'create_model_record',
            'update_model_status',
            'get_model',
            'get_model_by_fingerprint',
            'list_models',
            'delete_model',
            'delete_all_models'
        ]
        
        for method in required_methods:
            assert hasattr(db, method), f"Wrapper should have {method} method"
    
    @pytest.mark.asyncio
    async def test_sync_wrapper_basic_operations(self, db_tables):
        """Test basic CRUD operations through sync wrapper."""
        import os
        from training_service.sync_db_wrapper import SyncDBWrapper
        import training_service.pg_db as pg_db
        
        # Set test URL
        original_url = pg_db.POSTGRES_URL
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test'
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
    
    def test_multiple_processes_create_models(self):
        """Test that multiple processes can create models independently."""
        import os
        
        # Ensure test database URL is set
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test'
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
    
    def test_concurrent_model_updates(self):
        """Test concurrent updates from multiple processes."""
        import os
        
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test'
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
    
    def test_process_isolation(self):
        """Test that each process has isolated connection pool."""
        import os
        
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test'
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


@pytest.mark.skipif(
    not pytest.config.getoption("--run-slow", default=False),
    reason="Slow test, use --run-slow to run"
)
class TestSyncWrapperPerformance:
    """Performance tests for sync wrapper (slow, optional)."""
    
    def test_many_sequential_operations(self):
        """Test many sequential operations don't leak connections."""
        import os
        from training_service.sync_db_wrapper import db
        import training_service.pg_db as pg_db
        
        test_url = os.environ.get(
            'TEST_POSTGRES_URL',
            'postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test'
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

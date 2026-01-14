"""
Unit tests for training_service/pg_db.py
Tests async PostgreSQL database operations.
"""
import pytest
import json
import uuid
from datetime import datetime


@pytest.mark.asyncio
class TestPostgreSQLDatabaseLayer:
    """Test suite for PostgreSQL database operations."""
    
    async def test_ensure_tables(self, db_tables):
        """Test that all required tables are created."""
        async with db_tables.acquire() as conn:
            # Check models table exists
            models_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'models'
                )
            """)
            assert models_exists, "models table should exist"
            
            # Check features_log table exists
            features_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'features_log'
                )
            """)
            assert features_exists, "features_log table should exist"
            
            # Check simulation_history table exists
            sim_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'simulation_history'
                )
            """)
            assert sim_exists, "simulation_history table should exist"
    
    async def test_table_columns(self, db_tables):
        """Test that tables have all required columns."""
        async with db_tables.acquire() as conn:
            # Get columns for models table
            columns = await conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'models'
            """)
            
            column_names = [c['column_name'] for c in columns]
            
            # Check required fingerprint fields
            required_columns = [
                'id', 'algorithm', 'symbol', 'target_col', 'feature_cols',
                'hyperparameters', 'metrics', 'status', 'fingerprint',
                'timeframe', 'train_window', 'test_window', 'target_transform',
                'cv_folds', 'cv_strategy', 'alpha_grid', 'l1_ratio_grid',
                'context_symbols', 'regime_configs'
            ]
            
            for col in required_columns:
                assert col in column_names, f"Column '{col}' should exist in models table"
    
    async def test_indexes_created(self, db_tables):
        """Test that required indexes are created."""
        async with db_tables.acquire() as conn:
            # Get indexes
            indexes = await conn.fetch("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'models'
            """)
            
            index_names = [idx['indexname'] for idx in indexes]
            
            # Check for fingerprint index
            assert any('fingerprint' in idx for idx in index_names), \
                "Fingerprint index should exist"
            
            # Check for symbol index
            assert any('symbol' in idx for idx in index_names), \
                "Symbol index should exist"
    
    async def test_create_model_record(self, training_db, sample_model_data):
        """Test creating a model record with all fields."""
        model_id = sample_model_data['id']
        
        # Create model
        await training_db.create_model_record(sample_model_data)
        
        # Verify it was created
        model = await training_db.get_model(model_id)
        
        assert model is not None, "Model should be created"
        assert model['id'] == model_id
        assert model['algorithm'] == 'RandomForest'
        assert model['symbol'] == 'RDDT'
        assert model['status'] == 'completed'
        assert model['fingerprint'] == 'test_fingerprint_123'
    
    async def test_get_model_by_id(self, training_db, sample_model_data):
        """Test retrieving model by ID."""
        model_id = sample_model_data['id']
        
        # Create model
        await training_db.create_model_record(sample_model_data)
        
        # Get model
        model = await training_db.get_model(model_id)
        
        assert model is not None
        assert model['id'] == model_id
        assert model['algorithm'] == 'RandomForest'
    
    async def test_get_nonexistent_model(self, training_db):
        """Test getting a model that doesn't exist."""
        fake_id = str(uuid.uuid4())
        
        model = await training_db.get_model(fake_id)
        
        assert model is None, "Should return None for nonexistent model"
    
    async def test_get_model_by_fingerprint(self, training_db, sample_model_data):
        """Test fingerprint-based model lookup."""
        fingerprint = sample_model_data['fingerprint']
        
        # Create model
        await training_db.create_model_record(sample_model_data)
        
        # Get by fingerprint
        model = await training_db.get_model_by_fingerprint(fingerprint)
        
        assert model is not None
        assert model['fingerprint'] == fingerprint
        assert model['id'] == sample_model_data['id']
    
    async def test_update_model_status(self, training_db, sample_model_data):
        """Test updating model status."""
        model_id = sample_model_data['id']
        
        # Create model with initial status
        sample_model_data['status'] = 'preprocessing'
        await training_db.create_model_record(sample_model_data)
        
        # Update to training
        await training_db.update_model_status(
            model_id,
            status='training',
            error_message=None
        )
        
        model = await training_db.get_model(model_id)
        assert model['status'] == 'training'
        
        # Update to completed with metrics
        metrics = {"accuracy": 0.92, "f1": 0.89}
        await training_db.update_model_status(
            model_id,
            status='completed',
            metrics=metrics
        )
        
        model = await training_db.get_model(model_id)
        assert model['status'] == 'completed'
        
        # Parse metrics
        stored_metrics = model['metrics']
        if isinstance(stored_metrics, str):
            stored_metrics = json.loads(stored_metrics)
        assert stored_metrics['accuracy'] == 0.92
    
    async def test_update_model_status_with_error(self, training_db, sample_model_data):
        """Test updating model to failed status with error message."""
        model_id = sample_model_data['id']
        
        await training_db.create_model_record(sample_model_data)
        
        # Update to failed
        error_msg = "Training failed due to insufficient data"
        await training_db.update_model_status(
            model_id,
            status='failed',
            error_message=error_msg
        )
        
        model = await training_db.get_model(model_id)
        assert model['status'] == 'failed'
        assert error_msg in str(model.get('error_message', ''))
    
    async def test_list_models(self, training_db, sample_model_data):
        """Test listing multiple models."""
        # Create 3 models
        for i in range(3):
            data = sample_model_data.copy()
            data['id'] = str(uuid.uuid4())
            data['symbol'] = f'SYM{i}'
            await training_db.create_model_record(data)
        
        # List all models
        models = await training_db.list_models()
        
        assert len(models) >= 3, "Should return at least 3 models"
    
    async def test_list_models_with_limit(self, training_db, sample_model_data):
        """Test listing models with limit."""
        # Create 5 models
        for i in range(5):
            data = sample_model_data.copy()
            data['id'] = str(uuid.uuid4())
            await training_db.create_model_record(data)
        
        # List with limit
        models = await training_db.list_models(limit=2)
        
        assert len(models) == 2, "Should respect limit parameter"
    
    async def test_delete_model(self, training_db, sample_model_data):
        """Test deleting a model."""
        model_id = sample_model_data['id']
        
        # Create model
        await training_db.create_model_record(sample_model_data)
        
        # Verify it exists
        model = await training_db.get_model(model_id)
        assert model is not None
        
        # Delete it
        await training_db.delete_model(model_id)
        
        # Verify it's gone
        model = await training_db.get_model(model_id)
        assert model is None, "Model should be deleted"
    
    async def test_delete_all_models(self, training_db, sample_model_data):
        """Test deleting all models."""
        # Create multiple models
        for i in range(3):
            data = sample_model_data.copy()
            data['id'] = str(uuid.uuid4())
            await training_db.create_model_record(data)
        
        # Delete all
        await training_db.delete_all_models()
        
        # Verify all gone
        models = await training_db.list_models()
        assert len(models) == 0, "All models should be deleted"
    
    async def test_feature_importance_storage(self, db_tables, sample_model_data):
        """Test storing feature importance in features_log table."""
        model_id = sample_model_data['id']
        
        async with db_tables.acquire() as conn:
            # Create model
            await conn.execute("""
                INSERT INTO models (id, algorithm, symbol, status)
                VALUES ($1, $2, $3, $4)
            """, model_id, 'RandomForest', 'RDDT', 'completed')
            
            # Insert feature importance
            features = [
                ('sma_20', 0.35),
                ('rsi_14', 0.28),
                ('ema_10', 0.22)
            ]
            
            for feature_name, importance in features:
                await conn.execute("""
                    INSERT INTO features_log (model_id, feature_name, importance)
                    VALUES ($1, $2, $3)
                """, model_id, feature_name, importance)
            
            # Retrieve feature importance
            rows = await conn.fetch("""
                SELECT feature_name, importance
                FROM features_log
                WHERE model_id = $1
                ORDER BY importance DESC
            """, model_id)
            
            assert len(rows) == 3
            assert rows[0]['feature_name'] == 'sma_20'
            assert rows[0]['importance'] == 0.35
    
    async def test_cascade_delete_features(self, db_tables, sample_model_data):
        """Test that deleting a model cascades to features_log."""
        model_id = sample_model_data['id']
        
        async with db_tables.acquire() as conn:
            # Create model
            await conn.execute("""
                INSERT INTO models (id, algorithm, symbol, status)
                VALUES ($1, $2, $3, $4)
            """, model_id, 'RandomForest', 'RDDT', 'completed')
            
            # Insert feature
            await conn.execute("""
                INSERT INTO features_log (model_id, feature_name, importance)
                VALUES ($1, $2, $3)
            """, model_id, 'sma_20', 0.5)
            
            # Verify feature exists
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM features_log WHERE model_id = $1
            """, model_id)
            assert count == 1
            
            # Delete model
            await conn.execute("DELETE FROM models WHERE id = $1", model_id)
            
            # Verify features deleted (cascade)
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM features_log WHERE model_id = $1
            """, model_id)
            assert count == 0, "Features should be cascade deleted"
    
    async def test_simulation_history_save(self, simulation_db):
        """Test saving simulation history."""
        model_id = str(uuid.uuid4())
        ticker = "RDDT"
        stats = {
            "strategy_return_pct": 25.5,
            "total_trades": 50,
            "hit_rate_pct": 0.68,
            "sqn": 3.2
        }
        params = {"initial_cash": 10000}
        
        # Save simulation
        sim_id = await simulation_db.save_history(model_id, ticker, stats, params)
        
        assert sim_id is not None
        
        # Verify saved
        history = await simulation_db.get_history(limit=1)
        
        assert len(history) == 1
        assert history[0]['model_id'] == model_id
        assert history[0]['ticker'] == ticker
        assert history[0]['return_pct'] == 25.5
        assert history[0]['sqn'] == 3.2
    
    async def test_simulation_top_strategies(self, simulation_db):
        """Test retrieving top strategies by SQN."""
        # Create multiple simulations with different SQN values
        simulations = [
            ("model1", "RDDT", {"sqn": 3.5, "total_trades": 20}),
            ("model2", "AAPL", {"sqn": 2.1, "total_trades": 15}),
            ("model3", "MSFT", {"sqn": 4.2, "total_trades": 25}),
        ]
        
        for model_id, ticker, stats in simulations:
            stats.update({
                "strategy_return_pct": 10.0,
                "total_trades": stats["total_trades"],
                "hit_rate_pct": 0.6
            })
            await simulation_db.save_history(model_id, ticker, stats, {})
        
        # Get top strategies
        result = await simulation_db.get_top_strategies(limit=10)
        
        items = result['items']
        assert len(items) == 3
        
        # Verify sorted by SQN descending
        assert items[0]['sqn'] == 4.2  # model3
        assert items[1]['sqn'] == 3.5  # model1
        assert items[2]['sqn'] == 2.1  # model2
    
    async def test_simulation_pagination(self, simulation_db):
        """Test pagination of top strategies."""
        # Create 10 simulations
        for i in range(10):
            stats = {
                "strategy_return_pct": 10.0,
                "total_trades": 20,
                "hit_rate_pct": 0.6,
                "sqn": float(i)
            }
            await simulation_db.save_history(f"model{i}", "RDDT", stats, {})
        
        # Get first page
        page1 = await simulation_db.get_top_strategies(limit=3, offset=0)
        assert len(page1['items']) == 3
        assert page1['total'] == 10
        
        # Get second page
        page2 = await simulation_db.get_top_strategies(limit=3, offset=3)
        assert len(page2['items']) == 3
        
        # Verify different results
        assert page1['items'][0]['model_id'] != page2['items'][0]['model_id']
    
    async def test_delete_all_simulation_history(self, simulation_db):
        """Test deleting all simulation history."""
        # Create simulations
        for i in range(5):
            stats = {
                "strategy_return_pct": 10.0,
                "total_trades": 20,
                "hit_rate_pct": 0.6,
                "sqn": 2.0
            }
            await simulation_db.save_history(f"model{i}", "RDDT", stats, {})
        
        # Verify created
        history = await simulation_db.get_history(limit=10)
        assert len(history) == 5
        
        # Delete all
        success = await simulation_db.delete_all_history()
        assert success is True
        
        # Verify deleted
        history = await simulation_db.get_history(limit=10)
        assert len(history) == 0


@pytest.mark.asyncio
@pytest.mark.unit
class TestJSONBSerialization:
    """Regression tests for JSONB field serialization (Issue: asyncpg TypeError)."""
    
    async def test_create_model_with_json_string_fields(self, training_db):
        """
        Test creating a model with JSONB fields as JSON strings.
        
        Regression test for bug where json.loads() was called on JSON strings
        before passing to asyncpg, causing TypeError: expected str, got list.
        
        JSONB fields should accept JSON strings and convert them properly.
        """
        model_id = str(uuid.uuid4())
        model_data = {
            'id': model_id,
            'name': 'test-jsonb-serialization',
            'algorithm': 'RandomForest',
            'symbol': 'GOOGL',
            'target_col': 'close',
            'feature_cols': json.dumps([]),  # Empty list as JSON string
            'hyperparameters': json.dumps({'n_estimators': 100}),  # Dict as JSON string
            'metrics': json.dumps({'accuracy': 0.95}),  # Dict as JSON string
            'status': 'pending',
            'data_options': json.dumps({'train_window': 1000}),  # JSONB field as string
            'timeframe': '1m',
            'target_transform': 'log_return',
            'fingerprint': 'test_jsonb_fp'
        }
        
        # Should not raise TypeError
        await training_db.create_model_record(model_data)
        
        # Verify created correctly
        model = await training_db.get_model(model_id)
        assert model is not None
        assert model['id'] == model_id
        
        # Verify JSONB fields were stored and can be retrieved
        # asyncpg returns JSONB as Python objects
        assert model['feature_cols'] == []
        assert model['hyperparameters']['n_estimators'] == 100
        assert model['metrics']['accuracy'] == 0.95
        assert model['data_options']['train_window'] == 1000
    
    async def test_create_model_with_python_objects(self, training_db):
        """
        Test creating a model with JSONB fields as Python objects.
        
        The create_model_record function should handle both JSON strings
        AND Python dicts/lists, converting Python objects to JSON strings.
        """
        model_id = str(uuid.uuid4())
        model_data = {
            'id': model_id,
            'name': 'test-python-objects',
            'algorithm': 'ElasticNet',
            'symbol': 'AAPL',
            'target_col': 'close',
            'feature_cols': ['rsi_14', 'macd', 'bb_upper'],  # Python list
            'hyperparameters': {'alpha': 0.1, 'l1_ratio': 0.5},  # Python dict
            'metrics': {},  # Empty Python dict
            'status': 'pending',
            'data_options': {'train_window': 2000, 'test_window': 500},  # Python dict
            'alpha_grid': [0.001, 0.01, 0.1, 1.0],  # Python list
            'l1_ratio_grid': [0.1, 0.5, 0.9],  # Python list
            'timeframe': '5m',
            'target_transform': 'pct_change',
            'fingerprint': 'test_python_fp'
        }
        
        # Should not raise TypeError
        await training_db.create_model_record(model_data)
        
        # Verify created correctly
        model = await training_db.get_model(model_id)
        assert model is not None
        assert model['feature_cols'] == ['rsi_14', 'macd', 'bb_upper']
        assert model['hyperparameters']['alpha'] == 0.1
        assert model['alpha_grid'] == [0.001, 0.01, 0.1, 1.0]
        assert model['l1_ratio_grid'] == [0.1, 0.5, 0.9]
    
    async def test_create_model_with_none_jsonb_fields(self, training_db):
        """
        Test creating a model with None values for optional JSONB fields.
        
        asyncpg should handle None values for JSONB columns (stored as NULL).
        """
        model_id = str(uuid.uuid4())
        model_data = {
            'id': model_id,
            'name': 'test-none-fields',
            'algorithm': 'Lasso',
            'symbol': 'MSFT',
            'target_col': 'close',
            'feature_cols': json.dumps([]),
            'hyperparameters': json.dumps({}),
            'metrics': json.dumps({}),
            'status': 'pending',
            'data_options': None,  # None for optional JSONB field
            'alpha_grid': None,
            'l1_ratio_grid': None,
            'context_symbols': None,
            'timeframe': '1m',
            'fingerprint': 'test_none_fp'
        }
        
        # Should not raise error
        await training_db.create_model_record(model_data)
        
        # Verify created correctly with NULL values
        model = await training_db.get_model(model_id)
        assert model is not None
        assert model['data_options'] is None
        assert model['alpha_grid'] is None
        assert model['l1_ratio_grid'] is None
    
    async def test_create_model_mixed_serialization(self, training_db):
        """
        Test creating a model with mixed JSON strings and Python objects.
        
        Real-world scenario where some fields come as JSON strings (from API)
        and others as Python objects (from code).
        """
        model_id = str(uuid.uuid4())
        model_data = {
            'id': model_id,
            'name': 'test-mixed-types',
            'algorithm': 'Ridge',
            'symbol': 'QQQ',
            'target_col': 'close',
            'feature_cols': json.dumps(['feature_1', 'feature_2']),  # JSON string
            'hyperparameters': {'alpha': 50.0},  # Python dict
            'metrics': json.dumps({}),  # JSON string
            'status': 'pending',
            'data_options': None,  # None
            'alpha_grid': [1.0, 10.0, 50.0, 100.0],  # Python list
            'context_symbols': json.dumps(['MSFT', 'AAPL', 'GOOGL']),  # JSON string
            'timeframe': '15m',
            'fingerprint': 'test_mixed_fp'
        }
        
        # Should handle all types correctly
        await training_db.create_model_record(model_data)
        
        # Verify all fields stored correctly
        model = await training_db.get_model(model_id)
        assert model is not None
        assert model['feature_cols'] == ['feature_1', 'feature_2']
        assert model['hyperparameters']['alpha'] == 50.0
        assert model['alpha_grid'] == [1.0, 10.0, 50.0, 100.0]
        assert model['context_symbols'] == ['MSFT', 'AAPL', 'GOOGL']

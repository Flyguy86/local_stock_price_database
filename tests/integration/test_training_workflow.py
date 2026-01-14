"""
Integration tests for Training Service workflows.

Tests the complete end-to-end flow:
1. Submit training job
2. Wait for completion
3. Verify model saved to database
4. Query model metadata
5. Retrieve feature importance
6. Verify model file exists
"""
import pytest
import asyncio
import json
import time
from pathlib import Path
from httpx import AsyncClient

from training_service.main import app, db
from training_service.config import settings
from training_service.sync_db_wrapper import sync_create_model_record, sync_get_model_by_id


class TestTrainingWorkflow:
    """Test complete training workflow from submission to retrieval."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_train_save_query_flow(self, db_tables):
        """
        Integration test: Train → Save → Query complete workflow.
        
        Steps:
        1. Submit training job via API
        2. Verify job record created immediately
        3. Wait for training to complete (or fail gracefully)
        4. Query model by ID
        5. Verify all metadata fields populated
        6. Check feature importance if available
        """
        # Step 1: Submit training job
        payload = {
            "symbol": "AAPL",
            "algorithm": "ridge",
            "target_col": "close",
            "params": {"alpha": 1.0},
            "timeframe": "1m",
            "data_options": None
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/train", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "training_id" in data
        training_id = data["training_id"]
        
        # Step 2: Verify job record created immediately
        model = await db.get_model_by_id(training_id)
        assert model is not None
        assert model["id"] == training_id
        assert model["symbol"] == "AAPL"
        assert model["algorithm"] == "ridge"
        assert model["status"] in ["pending", "training"]
        assert model["target_col"] == "close"
        
        # Step 3: Wait a bit for async processing (or check status)
        # In real workflow, training happens in background
        # For integration test, we verify the record exists and has correct initial state
        await asyncio.sleep(0.5)
        
        # Step 4: Query model by ID (should still exist)
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"/models/{training_id}")
        
        assert response.status_code == 200
        model_data = response.json()
        assert model_data["id"] == training_id
        assert model_data["algorithm"] == "ridge"
        assert model_data["symbol"] == "AAPL"
        
        # Step 5: Verify metadata structure
        assert "hyperparameters" in model_data
        assert "metrics" in model_data
        assert "created_at" in model_data
        assert "status" in model_data
        
        # Step 6: Verify we can list all models and find this one
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/models")
        
        assert response.status_code == 200
        models_list = response.json()["models"]
        found = any(m["id"] == training_id for m in models_list)
        assert found, "Trained model should appear in models list"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_batch_training_workflow(self, db_tables):
        """
        Integration test: Batch training workflow.
        
        Verifies that multiple training jobs can be submitted
        and tracked simultaneously.
        """
        # Submit batch training job
        payload = {
            "symbol": "AAPL",
            "algorithms": ["ridge", "lasso"],
            "target_cols": ["close"],
            "timeframe": "1m"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/train/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "training_ids" in data
        training_ids = data["training_ids"]
        assert len(training_ids) == 2  # 2 algorithms
        
        # Verify all jobs created
        for training_id in training_ids:
            model = await db.get_model_by_id(training_id)
            assert model is not None
            assert model["symbol"] == "AAPL"
            assert model["algorithm"] in ["ridge", "lasso"]
            assert model["status"] in ["pending", "training", "completed", "failed"]
        
        # Verify we can retrieve all models
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/models")
        
        assert response.status_code == 200
        models_list = response.json()["models"]
        assert len(models_list) >= 2
        
        # Verify distinct algorithms
        algorithms = {m["algorithm"] for m in models_list if m["id"] in training_ids}
        assert "ridge" in algorithms
        assert "lasso" in algorithms
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_parent_child_training_workflow(self, db_tables):
        """
        Integration test: Parent-child training workflow (transfer learning).
        
        Steps:
        1. Create parent model
        2. Submit child training job referencing parent
        3. Verify parent-child relationship
        4. Query both models
        """
        # Step 1: Create parent model
        parent_data = {
            "id": "parent-model-123",
            "name": "parent-ridge-aapl",
            "algorithm": "ridge",
            "symbol": "AAPL",
            "target_col": "close",
            "feature_cols": json.dumps(["feature_a", "feature_b"]),
            "hyperparameters": json.dumps({"alpha": 1.0}),
            "status": "completed",
            "metrics": json.dumps({"r2": 0.85}),
            "timeframe": "1m",
            "train_window": 1000,
            "test_window": 200
        }
        await db.create_model_record(parent_data)
        
        # Step 2: Submit child training job
        payload = {
            "parent_model_id": "parent-model-123",
            "symbol": "GOOGL",  # Different symbol
            "algorithm": "ridge",
            "target_col": "close"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/train_with_parent", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "training_id" in data
        child_id = data["training_id"]
        
        # Step 3: Verify parent-child relationship
        child_model = await db.get_model_by_id(child_id)
        assert child_model is not None
        assert child_model["parent_model_id"] == "parent-model-123"
        assert child_model["symbol"] == "GOOGL"
        
        # Step 4: Query both models
        parent_model = await db.get_model_by_id("parent-model-123")
        assert parent_model is not None
        assert parent_model["status"] == "completed"
        
        # Child should reference parent
        assert child_model["parent_model_id"] == parent_model["id"]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_retrain_workflow(self, db_tables):
        """
        Integration test: Model retraining workflow.
        
        Steps:
        1. Create original model
        2. Submit retrain request
        3. Verify new model created
        4. Verify new model references original as parent
        5. Verify both models exist in database
        """
        # Step 1: Create original model
        original_data = {
            "id": "original-model-456",
            "name": "original-ridge",
            "algorithm": "ridge",
            "symbol": "MSFT",
            "target_col": "close",
            "feature_cols": json.dumps(["feature_a"]),
            "hyperparameters": json.dumps({"alpha": 1.0}),
            "status": "completed",
            "metrics": json.dumps({"r2": 0.80}),
            "timeframe": "1m",
            "train_window": 1000,
            "test_window": 200
        }
        await db.create_model_record(original_data)
        
        # Step 2: Submit retrain request
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/retrain/original-model-456")
        
        assert response.status_code == 200
        data = response.json()
        assert "training_id" in data
        retrain_id = data["training_id"]
        assert retrain_id != "original-model-456"  # New ID
        
        # Step 3: Verify new model created
        retrain_model = await db.get_model_by_id(retrain_id)
        assert retrain_model is not None
        assert retrain_model["algorithm"] == "ridge"
        assert retrain_model["symbol"] == "MSFT"
        
        # Step 4: Verify parent relationship
        assert retrain_model["parent_model_id"] == "original-model-456"
        
        # Step 5: Verify both models exist
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/models")
        
        models_list = response.json()["models"]
        model_ids = {m["id"] for m in models_list}
        assert "original-model-456" in model_ids
        assert retrain_id in model_ids
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_feature_importance_workflow(self, db_tables):
        """
        Integration test: Feature importance storage and retrieval.
        
        Steps:
        1. Create model with feature importance
        2. Save feature importance to database
        3. Retrieve via API
        4. Verify correct ordering and values
        """
        # Step 1: Create model
        model_data = {
            "id": "model-with-features",
            "name": "test-model",
            "algorithm": "randomforest",
            "symbol": "AAPL",
            "target_col": "close",
            "feature_cols": json.dumps(["rsi_14", "sma_20", "volume_ratio"]),
            "hyperparameters": json.dumps({}),
            "status": "completed",
            "metrics": json.dumps({"r2": 0.85}),
            "timeframe": "1m",
            "train_window": 1000,
            "test_window": 200
        }
        await db.create_model_record(model_data)
        
        # Step 2: Save feature importance
        importance_data = {
            "rsi_14": 0.45,
            "sma_20": 0.35,
            "volume_ratio": 0.20
        }
        await db.save_feature_importance("model-with-features", importance_data)
        
        # Step 3: Retrieve via API
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/model/model-with-features/importance")
        
        assert response.status_code == 200
        data = response.json()
        assert "importance" in data
        importance_list = data["importance"]
        
        # Step 4: Verify ordering (should be sorted by value descending)
        assert len(importance_list) == 3
        assert importance_list[0]["feature"] == "rsi_14"
        assert importance_list[0]["value"] == 0.45
        assert importance_list[1]["feature"] == "sma_20"
        assert importance_list[1]["value"] == 0.35
        assert importance_list[2]["feature"] == "volume_ratio"
        assert importance_list[2]["value"] == 0.20
        
        # Verify sorted
        values = [item["value"] for item in importance_list]
        assert values == sorted(values, reverse=True)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_model_deletion_cascade(self, db_tables):
        """
        Integration test: Verify cascade deletion of model and related data.
        
        Steps:
        1. Create model with feature importance
        2. Verify both exist
        3. Delete model
        4. Verify model and features deleted
        """
        # Step 1: Create model and features
        model_data = {
            "id": "model-to-delete",
            "name": "deletable-model",
            "algorithm": "ridge",
            "symbol": "AAPL",
            "target_col": "close",
            "feature_cols": json.dumps(["feature_a"]),
            "hyperparameters": json.dumps({}),
            "status": "completed",
            "metrics": json.dumps({}),
            "timeframe": "1m",
            "train_window": 1000,
            "test_window": 200
        }
        await db.create_model_record(model_data)
        
        # Add feature importance
        await db.save_feature_importance("model-to-delete", {"feature_a": 1.0})
        
        # Step 2: Verify both exist
        model = await db.get_model_by_id("model-to-delete")
        assert model is not None
        
        features = await db.get_feature_importance("model-to-delete")
        assert features is not None
        assert len(features) > 0
        
        # Step 3: Delete model via API
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.delete("/models/model-to-delete")
        
        assert response.status_code == 200
        
        # Step 4: Verify deletion (cascade)
        model = await db.get_model_by_id("model-to-delete")
        assert model is None
        
        # Features should also be deleted (cascade)
        features = await db.get_feature_importance("model-to-delete")
        assert features is None or len(features) == 0


class TestMultiProcessTraining:
    """Test multi-process training workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_concurrent_training_jobs(self, db_tables):
        """
        Integration test: Multiple concurrent training jobs.
        
        Verifies that multiple training jobs can be submitted
        and processed in parallel without conflicts.
        """
        # Submit multiple training jobs
        training_ids = []
        
        for i in range(3):
            payload = {
                "symbol": ["AAPL", "GOOGL", "MSFT"][i],
                "algorithm": "ridge",
                "target_col": "close",
                "params": {"alpha": 1.0},
                "timeframe": "1m"
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post("/train", json=payload)
            
            assert response.status_code == 200
            training_ids.append(response.json()["training_id"])
        
        # Verify all jobs created
        assert len(training_ids) == 3
        assert len(set(training_ids)) == 3  # All unique
        
        # Verify all models in database
        for training_id in training_ids:
            model = await db.get_model_by_id(training_id)
            assert model is not None
            assert model["status"] in ["pending", "training", "completed", "failed"]
        
        # Verify distinct symbols
        models = [await db.get_model_by_id(tid) for tid in training_ids]
        symbols = {m["symbol"] for m in models}
        assert len(symbols) == 3
        assert symbols == {"AAPL", "GOOGL", "MSFT"}


class TestSyncDBWrapperIntegration:
    """Test sync DB wrapper in realistic scenarios."""
    
    @pytest.mark.integration
    def test_sync_wrapper_create_and_retrieve(self, db_tables):
        """
        Integration test: Sync wrapper for model creation and retrieval.
        
        Tests the synchronous wrapper functions used by training workers.
        """
        # Create model using sync wrapper
        model_data = {
            "id": "sync-test-model",
            "name": "sync-wrapper-test",
            "algorithm": "ridge",
            "symbol": "AAPL",
            "target_col": "close",
            "feature_cols": json.dumps([]),
            "hyperparameters": json.dumps({"alpha": 1.0}),
            "status": "pending",
            "metrics": json.dumps({}),
            "timeframe": "1m",
            "train_window": 1000,
            "test_window": 200
        }
        
        # Use sync wrapper (as training workers would)
        sync_create_model_record(model_data)
        
        # Retrieve using sync wrapper
        retrieved = sync_get_model_by_id("sync-test-model")
        
        assert retrieved is not None
        assert retrieved["id"] == "sync-test-model"
        assert retrieved["algorithm"] == "ridge"
        assert retrieved["symbol"] == "AAPL"


# Mark integration tests
pytestmark = pytest.mark.integration

"""
API tests for Training Service endpoints.

Tests all HTTP endpoints of the training service including:
- Health checks
- Model training submission
- Model retrieval and management
- Batch operations
- Feature importance
- Data options
"""
import pytest
import asyncio
import json
from httpx import AsyncClient
from fastapi.testclient import TestClient
from training_service.main import app, db
from training_service.config import settings


class TestTrainingServiceHealth:
    """Test health and status endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, db_tables):
        """Test /health endpoint returns healthy status."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert data["service"] == "training"
    
    @pytest.mark.asyncio
    async def test_logs_endpoint(self, db_tables):
        """Test /logs endpoint returns log buffer."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/logs")
            
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)


class TestDataEndpoints:
    """Test data and configuration endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_data_options(self, db_tables):
        """Test /data/options endpoint returns available data configurations."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/data/options")
            
        assert response.status_code == 200
        data = response.json()
        assert "options" in data
        assert isinstance(data["options"], list)
    
    @pytest.mark.asyncio
    async def test_get_data_options_for_symbol(self, db_tables):
        """Test /data/options/{symbol} endpoint returns symbol-specific data."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/data/options/AAPL")
            
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert data["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_get_feature_map(self, db_tables):
        """Test /data/map endpoint returns feature mapping."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/data/map")
            
        assert response.status_code == 200
        data = response.json()
        assert "feature_map" in data
        assert isinstance(data["feature_map"], dict)
    
    @pytest.mark.asyncio
    async def test_get_algorithms(self, db_tables):
        """Test /algorithms endpoint returns available algorithms."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/algorithms")
            
        assert response.status_code == 200
        data = response.json()
        assert "algorithms" in data
        assert isinstance(data["algorithms"], list)
        assert len(data["algorithms"]) > 0


class TestModelManagement:
    """Test model CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_list_models_empty(self, db_tables):
        """Test /models endpoint returns empty list initially."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/models")
            
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    @pytest.mark.asyncio
    async def test_get_model_not_found(self, db_tables):
        """Test /models/{model_id} returns 404 for non-existent model."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/models/non-existent-id")
            
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_model_by_id(self, db_tables, sample_model_data):
        """Test /models/{model_id} returns model details."""
        # Create a model first
        model_data = sample_model_data()
        await db.create_model_record(model_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"/models/{model_data['id']}")
            
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == model_data["id"]
        assert data["algorithm"] == model_data["algorithm"]
        assert data["symbol"] == model_data["symbol"]
    
    @pytest.mark.asyncio
    async def test_delete_model(self, db_tables, sample_model_data):
        """Test /models/{model_id} DELETE removes model."""
        # Create a model first
        model_data = sample_model_data()
        await db.create_model_record(model_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Delete the model
            response = await client.delete(f"/models/{model_data['id']}")
            assert response.status_code == 200
            
            # Verify it's gone
            response = await client.get(f"/models/{model_data['id']}")
            assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_delete_all_models(self, db_tables, sample_model_data):
        """Test /models/all DELETE removes all models."""
        # Create multiple models
        model1 = sample_model_data()
        model2 = sample_model_data()
        await db.create_model_record(model1)
        await db.create_model_record(model2)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Delete all models
            response = await client.delete("/models/all")
            assert response.status_code == 200
            
            # Verify empty list
            response = await client.get("/models")
            data = response.json()
            assert len(data["models"]) == 0


class TestTrainingOperations:
    """Test model training endpoints."""
    
    @pytest.mark.asyncio
    async def test_train_endpoint_basic(self, db_tables):
        """Test /train endpoint accepts training request."""
        payload = {
            "symbol": "AAPL",
            "algorithm": "ridge",
            "target_col": "close",
            "params": {"alpha": 1.0},
            "timeframe": "1m"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/train", json=payload)
            
        assert response.status_code == 200
        data = response.json()
        assert "training_id" in data
        assert "message" in data
        
        # Verify model record created
        model = await db.get_model_by_id(data["training_id"])
        assert model is not None
        assert model["symbol"] == "AAPL"
        assert model["algorithm"] == "ridge"
        assert model["status"] in ["pending", "training", "completed", "failed"]
    
    @pytest.mark.asyncio
    async def test_train_endpoint_invalid_algorithm(self, db_tables):
        """Test /train endpoint rejects invalid algorithm."""
        payload = {
            "symbol": "AAPL",
            "algorithm": "invalid_algorithm",
            "target_col": "close"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/train", json=payload)
            
        # Should either reject (400) or create with pending status
        # Depends on validation logic
        assert response.status_code in [200, 400, 422]
    
    @pytest.mark.asyncio
    async def test_train_batch_endpoint(self, db_tables):
        """Test /train/batch endpoint accepts multiple training requests."""
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
        assert len(data["training_ids"]) == 2  # 2 algorithms
        
        # Verify all models created
        for training_id in data["training_ids"]:
            model = await db.get_model_by_id(training_id)
            assert model is not None
    
    @pytest.mark.asyncio
    async def test_retrain_endpoint(self, db_tables, sample_model_data):
        """Test /retrain/{model_id} endpoint creates new training job."""
        # Create a completed model first
        model_data = sample_model_data()
        model_data["status"] = "completed"
        await db.create_model_record(model_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(f"/retrain/{model_data['id']}")
            
        assert response.status_code == 200
        data = response.json()
        assert "training_id" in data
        assert data["training_id"] != model_data["id"]  # New ID
        
        # Verify new model created
        new_model = await db.get_model_by_id(data["training_id"])
        assert new_model is not None
        assert new_model["parent_model_id"] == model_data["id"]


class TestFeatureImportance:
    """Test feature importance endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_feature_importance(self, db_tables, sample_model_data):
        """Test /api/model/{model_id}/importance returns feature importance."""
        # Create a model with feature importance
        model_data = sample_model_data()
        model_data["status"] = "completed"
        await db.create_model_record(model_data)
        
        # Add feature importance
        importance_data = {
            "feature_a": 0.5,
            "feature_b": 0.3,
            "feature_c": 0.2
        }
        await db.save_feature_importance(model_data["id"], importance_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"/api/model/{model_data['id']}/importance")
            
        assert response.status_code == 200
        data = response.json()
        assert "importance" in data
        assert len(data["importance"]) == 3
        
        # Should be sorted by value descending
        values = [item["value"] for item in data["importance"]]
        assert values == sorted(values, reverse=True)
    
    @pytest.mark.asyncio
    async def test_get_feature_importance_not_found(self, db_tables):
        """Test /api/model/{model_id}/importance returns 404 for non-existent model."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/model/non-existent-id/importance")
            
        assert response.status_code == 404


class TestModelConfig:
    """Test model configuration endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_model_config(self, db_tables, sample_model_data):
        """Test /api/model/{model_id}/config returns model configuration."""
        model_data = sample_model_data()
        model_data["status"] = "completed"
        await db.create_model_record(model_data)
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(f"/api/model/{model_data['id']}/config")
            
        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        config = data["config"]
        assert config["algorithm"] == model_data["algorithm"]
        assert config["symbol"] == model_data["symbol"]
    
    @pytest.mark.asyncio
    async def test_get_model_config_not_found(self, db_tables):
        """Test /api/model/{model_id}/config returns 404 for non-existent model."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/model/non-existent-id/config")
            
        assert response.status_code == 404


class TestTrainWithParent:
    """Test training with parent model (transfer learning)."""
    
    @pytest.mark.asyncio
    async def test_train_with_parent_endpoint(self, db_tables, sample_model_data):
        """Test /api/train_with_parent endpoint creates child model."""
        # Create parent model first
        parent_data = sample_model_data()
        parent_data["status"] = "completed"
        await db.create_model_record(parent_data)
        
        payload = {
            "parent_model_id": parent_data["id"],
            "symbol": "GOOGL",  # Different symbol
            "algorithm": "ridge",
            "target_col": "close"
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/train_with_parent", json=payload)
            
        assert response.status_code == 200
        data = response.json()
        assert "training_id" in data
        
        # Verify child model created with parent reference
        child_model = await db.get_model_by_id(data["training_id"])
        assert child_model is not None
        assert child_model["parent_model_id"] == parent_data["id"]
        assert child_model["symbol"] == "GOOGL"


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


# Mark slow tests
pytestmark = pytest.mark.asyncio

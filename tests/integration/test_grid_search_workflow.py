"""
Integration tests for Grid Search functionality.

Tests the complete grid search workflow:
1. Grid search saves all individual models with parent_model_id
2. Models have is_grid_member flag set correctly
3. Orchestrator can query and count grid models
4. save_all_grid_models parameter flows through entire chain
"""
import pytest
import asyncio
import json
import numpy as np
import pandas as pd
from pathlib import Path
from httpx import AsyncClient
from unittest.mock import patch, MagicMock, AsyncMock

from training_service.main import app, db
from training_service.trainer import train_model_task
from training_service.sync_db_wrapper import sync_get_model_by_id


class TestGridSearchWorkflow:
    """Test complete grid search workflow from training to model counting."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_grid_models_have_correct_parent_id(self, db_tables, mock_features):
        """
        Verify grid models are saved with parent_model_id = training_id.
        
        This is the core bug we fixed - models were being saved with parent_model_id=None
        instead of parent_model_id=training_id, causing orchestrator to find 0 models.
        """
        training_id = str(uuid.uuid4())
        
        # Mock feature loading to return synthetic data
        with patch('training_service.data.load_training_data', return_value=mock_features):
            # Call train_model_task with grid search parameters
            result = await asyncio.to_thread(
                train_model_task,
                training_id=training_id,
                symbol="AAPL",
                algorithm="elasticnet",
                target_col="close",
                timeframe="1m",
                target_transform="log_return",
                params=None,
                data_options=None,
                parent_model_id=None,
                grid_search=True,
                alpha_grid=[0.01, 0.1],
                l1_ratio_grid=[0.3, 0.7],
                save_all_grid_models=True
            )
        
        # Verify parent model was created
        parent = await db.get_model_by_id(training_id)
        assert parent is not None
        assert parent["id"] == training_id
        assert parent["algorithm"] == "elasticnet"
        assert parent.get("is_grid_member", False) is False  # Parent is not a grid member
        
        # Query all models from database
        all_models = await db.list_models()
        
        # Filter for grid members with this parent_id
        grid_models = [
            m for m in all_models
            if m.get("parent_model_id") == training_id and m.get("is_grid_member") is True
        ]
        
        # Should have 2 alphas × 2 l1_ratios = 4 grid models
        expected_count = 4
        assert len(grid_models) == expected_count, (
            f"Expected {expected_count} grid models, found {len(grid_models)}. "
            f"This is the bug we fixed - models should have parent_model_id={training_id}"
        )
        
        # Verify each grid model has correct parent linkage
        for model in grid_models:
            assert model["parent_model_id"] == training_id, (
                f"Grid model {model['id']} has parent_model_id={model.get('parent_model_id')}, "
                f"expected {training_id}"
            )
            assert model["is_grid_member"] is True
            assert model["algorithm"] == "elasticnet"
            assert model["symbol"] == "AAPL"
            
            # Verify hyperparameters exist and are different across models
            assert model.get("hyperparameters") is not None
            params = json.loads(model["hyperparameters"]) if isinstance(model["hyperparameters"], str) else model["hyperparameters"]
            assert "alpha" in params
            assert "l1_ratio" in params
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_is_grid_member_flag_set_correctly(self, db_tables, mock_features):
        """
        Verify is_grid_member flag is True for grid models, False for parent.
        """
        training_id = str(uuid.uuid4())
        
        with patch('training_service.data.load_training_data', return_value=mock_features):
            # Train with small grid
            result = await asyncio.to_thread(
                train_model_task,
                training_id=training_id,
                symbol="MSFT",
                algorithm="elasticnet",
                target_col="close",
                timeframe="1m",
                target_transform="log_return",
                params=None,
                data_options=None,
                parent_model_id=None,
                grid_search=True,
                alpha_grid=[0.1],
                l1_ratio_grid=[0.5],
                save_all_grid_models=True
            )
        
        # Get parent model
        parent = await db.get_model_by_id(training_id)
        assert parent.get("is_grid_member", False) is False, "Parent model should have is_grid_member=False"
        
        # Get grid models
        all_models = await db.list_models()
        grid_models = [m for m in all_models if m.get("parent_model_id") == training_id]
        
        # All grid models should have is_grid_member=True
        for model in grid_models:
            assert model["is_grid_member"] is True, (
                f"Grid model {model['id']} should have is_grid_member=True"
            )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_orchestrator_can_count_grid_models(self, db_tables, mock_features):
        """
        Test the exact query logic used by orchestrator to count grid models.
        
        This simulates orchestrator's behavior:
        1. Training completes with model_id
        2. Orchestrator queries: WHERE parent_model_id=model_id AND is_grid_member=True
        3. Count should match grid size
        """
        training_id = str(uuid.uuid4())
        
        # Define grid parameters
        alpha_grid = [0.001, 0.01, 0.1]
        l1_ratio_grid = [0.3, 0.5]
        expected_count = len(alpha_grid) * len(l1_ratio_grid)  # 3 × 2 = 6
        
        with patch('training_service.data.load_training_data', return_value=mock_features):
            # Train grid
            result = await asyncio.to_thread(
                train_model_task,
                training_id=training_id,
                symbol="NVDA",
                algorithm="elasticnet",
                target_col="close",
                timeframe="1m",
                target_transform="log_return",
                params=None,
                data_options=None,
                parent_model_id=None,
                grid_search=True,
                alpha_grid=alpha_grid,
                l1_ratio_grid=l1_ratio_grid,
                save_all_grid_models=True
            )
        
        # Simulate orchestrator's query (from main.py line 1280-1303)
        all_models = await db.list_models()
        grid_models = [
            m for m in all_models
            if m.get("parent_model_id") == training_id and m.get("is_grid_member") is True
        ]
        models_trained = len(grid_models)
        
        assert models_trained == expected_count, (
            f"Orchestrator found {models_trained} models, expected {expected_count}. "
            f"Query: parent_model_id={training_id} AND is_grid_member=True"
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_save_all_grid_models_parameter_flow(self, db_tables, mock_features):
        """
        Verify save_all_grid_models=False skips individual model saving.
        
        When save_all_grid_models=False, only parent grid model should be saved,
        not individual parameter combinations.
        """
        training_id = str(uuid.uuid4())
        
        with patch('training_service.data.load_training_data', return_value=mock_features):
            # Train with save_all_grid_models=False
            result = await asyncio.to_thread(
                train_model_task,
                training_id=training_id,
                symbol="QQQ",
                algorithm="elasticnet",
                target_col="close",
                timeframe="1m",
                target_transform="log_return",
                params=None,
                data_options=None,
                parent_model_id=None,
                grid_search=True,
                alpha_grid=[0.01, 0.1],
                l1_ratio_grid=[0.3, 0.7],
                save_all_grid_models=False  # Don't save individual models
            )
        
        # Parent should exist
        parent = await db.get_model_by_id(training_id)
        assert parent is not None
        
        # Should have NO grid members
        all_models = await db.list_models()
        grid_models = [
            m for m in all_models
            if m.get("parent_model_id") == training_id and m.get("is_grid_member") is True
        ]
        
        assert len(grid_models) == 0, (
            f"save_all_grid_models=False should not save individual models, found {len(grid_models)}"
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_grid_models_have_unique_hyperparameters(self, db_tables, mock_features):
        """
        Verify each grid model has unique hyperparameter combinations.
        """
        training_id = str(uuid.uuid4())
        
        alpha_grid = [0.01, 0.1, 1.0]
        l1_ratio_grid = [0.3, 0.7]
        
        with patch('training_service.data.load_training_data', return_value=mock_features):
            result = await asyncio.to_thread(
                train_model_task,
                training_id=training_id,
                symbol="GOOGL",
                algorithm="elasticnet",
                target_col="close",
                timeframe="1m",
                target_transform="log_return",
                params=None,
                data_options=None,
                parent_model_id=None,
                grid_search=True,
                alpha_grid=alpha_grid,
                l1_ratio_grid=l1_ratio_grid,
                save_all_grid_models=True
            )
        
        # Get all grid models
        all_models = await db.list_models()
        grid_models = [
            m for m in all_models
            if m.get("parent_model_id") == training_id and m.get("is_grid_member") is True
        ]
        
        # Extract hyperparameters from each model
        param_sets = []
        for model in grid_models:
            params = json.loads(model["hyperparameters"]) if isinstance(model["hyperparameters"], str) else model["hyperparameters"]
            param_tuple = (params["alpha"], params["l1_ratio"])
            param_sets.append(param_tuple)
        
        # All combinations should be unique
        assert len(param_sets) == len(set(param_sets)), (
            "Grid models should have unique hyperparameter combinations"
        )
        
        # Verify we have all expected combinations
        expected_combinations = {(a, r) for a in alpha_grid for r in l1_ratio_grid}
        actual_combinations = set(param_sets)
        assert actual_combinations == expected_combinations, (
            f"Expected combinations: {expected_combinations}, got: {actual_combinations}"
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_grid_models_have_metrics(self, db_tables, mock_features):
        """
        Verify grid models have valid metrics (MSE, R2, etc).
        """
        training_id = str(uuid.uuid4())
        
        with patch('training_service.data.load_training_data', return_value=mock_features):
            result = await asyncio.to_thread(
                train_model_task,
                training_id=training_id,
                symbol="XOM",
                algorithm="elasticnet",
                target_col="close",
                timeframe="1m",
                target_transform="log_return",
                params=None,
                data_options=None,
                parent_model_id=None,
                grid_search=True,
                alpha_grid=[0.1],
                l1_ratio_grid=[0.5],
                save_all_grid_models=True
            )
        
        # Get grid models
        all_models = await db.list_models()
        grid_models = [
            m for m in all_models
            if m.get("parent_model_id") == training_id and m.get("is_grid_member") is True
        ]
        
        # Each model should have metrics
        for model in grid_models:
            assert model.get("metrics") is not None, f"Model {model['id']} missing metrics"
            
            metrics = json.loads(model["metrics"]) if isinstance(model["metrics"], str) else model["metrics"]
            
            # Should have standard regression metrics
            assert "mse" in metrics or "test_mse" in metrics, "Missing MSE metric"
            assert "r2" in metrics or "test_r2" in metrics, "Missing R2 metric"


import uuid


@pytest.fixture
def mock_features():
    """
    Generate minimal mock feature data for tests.
    
    Returns:
        DataFrame with synthetic features and target column
    """
    np.random.seed(42)
    
    # Generate synthetic features and target
    n_samples = 200
    n_features = 10
    
    # Create feature columns
    feature_data = {f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)}
    feature_data["close"] = np.random.randn(n_samples)  # Target column
    feature_data["ts"] = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    
    df = pd.DataFrame(feature_data)
    
    return df

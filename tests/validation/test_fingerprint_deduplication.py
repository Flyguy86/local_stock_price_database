"""
Validation tests for model fingerprint deduplication.

Tests that the fingerprinting system correctly identifies duplicate models
and prevents redundant training. The fingerprint is computed from:
- features, hyperparameters, target_transform, symbol, target_col
- timeframe, train_window, test_window, context_symbols
- cv_folds, cv_strategy, alpha_grid, l1_ratio_grid, regime_configs
"""
import pytest
import json
from training_service.main import db
from training_service.pg_db import compute_fingerprint


class TestFingerprintGeneration:
    """Test fingerprint computation logic."""
    
    @pytest.mark.asyncio
    async def test_identical_configs_same_fingerprint(self, db_tables):
        """
        Validation: Identical model configurations produce identical fingerprints.
        
        Ensures that two models with exactly the same configuration
        generate the same fingerprint hash.
        """
        config1 = {
            "features": ["rsi_14", "sma_20"],
            "hyperparameters": {"alpha": 1.0},
            "target_transform": "log_return",
            "symbol": "AAPL",
            "target_col": "close",
            "timeframe": "1m",
            "train_window": 1000,
            "test_window": 200,
            "context_symbols": ["QQQ", "VIX"],
            "cv_folds": 5,
            "cv_strategy": "time_series",
            "alpha_grid": [0.1, 1.0, 10.0],
            "l1_ratio_grid": [0.1, 0.5, 0.9],
            "regime_configs": {"vix_threshold": 20}
        }
        
        config2 = config1.copy()
        
        fingerprint1 = compute_fingerprint(config1)
        fingerprint2 = compute_fingerprint(config2)
        
        assert fingerprint1 == fingerprint2
        assert len(fingerprint1) == 64  # SHA256 hex digest
    
    @pytest.mark.asyncio
    async def test_different_features_different_fingerprint(self, db_tables):
        """
        Validation: Different features produce different fingerprints.
        
        Changing the feature list should result in a different fingerprint.
        """
        config_base = {
            "features": ["rsi_14", "sma_20"],
            "hyperparameters": {"alpha": 1.0},
            "target_transform": "log_return",
            "symbol": "AAPL",
            "target_col": "close",
            "timeframe": "1m",
            "train_window": 1000,
            "test_window": 200
        }
        
        config_diff = config_base.copy()
        config_diff["features"] = ["rsi_14", "sma_50"]  # Different feature
        
        fingerprint_base = compute_fingerprint(config_base)
        fingerprint_diff = compute_fingerprint(config_diff)
        
        assert fingerprint_base != fingerprint_diff
    
    @pytest.mark.asyncio
    async def test_different_hyperparams_different_fingerprint(self, db_tables):
        """
        Validation: Different hyperparameters produce different fingerprints.
        
        Changing hyperparameters should result in a different fingerprint.
        """
        config_base = {
            "features": ["rsi_14"],
            "hyperparameters": {"alpha": 1.0},
            "target_transform": "log_return",
            "symbol": "AAPL",
            "target_col": "close"
        }
        
        config_diff = config_base.copy()
        config_diff["hyperparameters"] = {"alpha": 10.0}  # Different alpha
        
        fingerprint_base = compute_fingerprint(config_base)
        fingerprint_diff = compute_fingerprint(config_diff)
        
        assert fingerprint_base != fingerprint_diff
    
    @pytest.mark.asyncio
    async def test_different_symbol_different_fingerprint(self, db_tables):
        """
        Validation: Different symbols produce different fingerprints.
        
        Same model for different symbols should have different fingerprints.
        """
        config_aapl = {
            "features": ["rsi_14"],
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL",
            "target_col": "close"
        }
        
        config_googl = config_aapl.copy()
        config_googl["symbol"] = "GOOGL"
        
        fingerprint_aapl = compute_fingerprint(config_aapl)
        fingerprint_googl = compute_fingerprint(config_googl)
        
        assert fingerprint_aapl != fingerprint_googl
    
    @pytest.mark.asyncio
    async def test_order_independent_fingerprint(self, db_tables):
        """
        Validation: Feature order doesn't affect fingerprint.
        
        Features should be sorted before hashing to ensure
        [A, B, C] and [C, A, B] produce the same fingerprint.
        """
        config1 = {
            "features": ["rsi_14", "sma_20", "volume_ratio"],
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL"
        }
        
        config2 = {
            "features": ["volume_ratio", "rsi_14", "sma_20"],  # Different order
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL"
        }
        
        fingerprint1 = compute_fingerprint(config1)
        fingerprint2 = compute_fingerprint(config2)
        
        assert fingerprint1 == fingerprint2


class TestFingerprintDeduplication:
    """Test fingerprint-based deduplication in database."""
    
    @pytest.mark.asyncio
    async def test_lookup_by_fingerprint_not_found(self, db_tables):
        """
        Validation: Lookup returns None for non-existent fingerprint.
        """
        config = {
            "features": ["rsi_14"],
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL",
            "target_col": "close"
        }
        
        fingerprint = compute_fingerprint(config)
        existing = await db.get_model_by_fingerprint(fingerprint)
        
        assert existing is None
    
    @pytest.mark.asyncio
    async def test_lookup_by_fingerprint_found(self, db_tables, sample_model_data):
        """
        Validation: Lookup finds existing model by fingerprint.
        
        After creating a model, we should be able to retrieve it
        by its fingerprint.
        """
        # Create model with known fingerprint
        model_data = sample_model_data()
        model_data["feature_cols"] = json.dumps(["rsi_14", "sma_20"])
        model_data["hyperparameters"] = json.dumps({"alpha": 1.0})
        model_data["target_transform"] = "log_return"
        
        # Compute and set fingerprint
        config = {
            "features": json.loads(model_data["feature_cols"]),
            "hyperparameters": json.loads(model_data["hyperparameters"]),
            "target_transform": model_data["target_transform"],
            "symbol": model_data["symbol"],
            "target_col": model_data["target_col"],
            "timeframe": model_data["timeframe"],
            "train_window": model_data["train_window"],
            "test_window": model_data["test_window"]
        }
        fingerprint = compute_fingerprint(config)
        model_data["fingerprint"] = fingerprint
        
        # Save model
        await db.create_model_record(model_data)
        
        # Lookup by fingerprint
        found = await db.get_model_by_fingerprint(fingerprint)
        
        assert found is not None
        assert found["id"] == model_data["id"]
        assert found["fingerprint"] == fingerprint
    
    @pytest.mark.asyncio
    async def test_duplicate_prevention(self, db_tables, sample_model_data):
        """
        Validation: System prevents duplicate model training.
        
        When attempting to create a model with an existing fingerprint,
        the system should detect the duplicate.
        """
        # Create first model
        model1 = sample_model_data()
        model1["id"] = "model-original"
        model1["feature_cols"] = json.dumps(["rsi_14", "sma_20"])
        model1["hyperparameters"] = json.dumps({"alpha": 1.0})
        model1["symbol"] = "AAPL"
        model1["target_col"] = "close"
        model1["target_transform"] = "log_return"
        
        config = {
            "features": ["rsi_14", "sma_20"],
            "hyperparameters": {"alpha": 1.0},
            "target_transform": "log_return",
            "symbol": "AAPL",
            "target_col": "close",
            "timeframe": model1["timeframe"],
            "train_window": model1["train_window"],
            "test_window": model1["test_window"]
        }
        fingerprint = compute_fingerprint(config)
        model1["fingerprint"] = fingerprint
        
        await db.create_model_record(model1)
        
        # Attempt to create second model with same fingerprint
        existing = await db.get_model_by_fingerprint(fingerprint)
        
        assert existing is not None
        assert existing["id"] == "model-original"
        
        # This proves the system can detect duplicates before creating
    
    @pytest.mark.asyncio
    async def test_different_configs_no_collision(self, db_tables, sample_model_data):
        """
        Validation: Different configurations don't collide.
        
        Models with different configurations should have different
        fingerprints and not be considered duplicates.
        """
        # Create model 1: AAPL with alpha=1.0
        model1 = sample_model_data()
        model1["id"] = "model-1"
        model1["symbol"] = "AAPL"
        model1["hyperparameters"] = json.dumps({"alpha": 1.0})
        
        config1 = {
            "features": json.loads(model1["feature_cols"]),
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL",
            "target_col": model1["target_col"]
        }
        fingerprint1 = compute_fingerprint(config1)
        model1["fingerprint"] = fingerprint1
        
        await db.create_model_record(model1)
        
        # Create model 2: AAPL with alpha=10.0 (different hyperparameter)
        model2 = sample_model_data()
        model2["id"] = "model-2"
        model2["symbol"] = "AAPL"
        model2["hyperparameters"] = json.dumps({"alpha": 10.0})
        
        config2 = {
            "features": json.loads(model2["feature_cols"]),
            "hyperparameters": {"alpha": 10.0},
            "symbol": "AAPL",
            "target_col": model2["target_col"]
        }
        fingerprint2 = compute_fingerprint(config2)
        model2["fingerprint"] = fingerprint2
        
        await db.create_model_record(model2)
        
        # Verify different fingerprints
        assert fingerprint1 != fingerprint2
        
        # Verify both exist
        found1 = await db.get_model_by_fingerprint(fingerprint1)
        found2 = await db.get_model_by_fingerprint(fingerprint2)
        
        assert found1["id"] == "model-1"
        assert found2["id"] == "model-2"
    
    @pytest.mark.asyncio
    async def test_multiple_symbols_same_config(self, db_tables, sample_model_data):
        """
        Validation: Same configuration for different symbols creates unique fingerprints.
        
        Training the same model for AAPL and GOOGL should create
        two different fingerprints (symbol is part of fingerprint).
        """
        # Model for AAPL
        model_aapl = sample_model_data()
        model_aapl["id"] = "model-aapl"
        model_aapl["symbol"] = "AAPL"
        model_aapl["hyperparameters"] = json.dumps({"alpha": 1.0})
        
        config_aapl = {
            "features": json.loads(model_aapl["feature_cols"]),
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL",
            "target_col": model_aapl["target_col"]
        }
        fingerprint_aapl = compute_fingerprint(config_aapl)
        model_aapl["fingerprint"] = fingerprint_aapl
        
        await db.create_model_record(model_aapl)
        
        # Model for GOOGL (same config, different symbol)
        model_googl = sample_model_data()
        model_googl["id"] = "model-googl"
        model_googl["symbol"] = "GOOGL"
        model_googl["hyperparameters"] = json.dumps({"alpha": 1.0})
        
        config_googl = {
            "features": json.loads(model_googl["feature_cols"]),
            "hyperparameters": {"alpha": 1.0},
            "symbol": "GOOGL",
            "target_col": model_googl["target_col"]
        }
        fingerprint_googl = compute_fingerprint(config_googl)
        model_googl["fingerprint"] = fingerprint_googl
        
        await db.create_model_record(model_googl)
        
        # Verify different fingerprints
        assert fingerprint_aapl != fingerprint_googl
        
        # Verify both exist independently
        assert await db.get_model_by_fingerprint(fingerprint_aapl) is not None
        assert await db.get_model_by_fingerprint(fingerprint_googl) is not None


class TestFingerprintEdgeCases:
    """Test edge cases in fingerprint handling."""
    
    @pytest.mark.asyncio
    async def test_null_optional_fields(self, db_tables):
        """
        Validation: Null optional fields handled correctly in fingerprint.
        
        Optional fields like context_symbols, cv_folds, etc. may be None.
        Fingerprint should handle these gracefully.
        """
        config1 = {
            "features": ["rsi_14"],
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL",
            "target_col": "close",
            "context_symbols": None,
            "cv_folds": None,
            "alpha_grid": None
        }
        
        config2 = {
            "features": ["rsi_14"],
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL",
            "target_col": "close"
            # Optional fields omitted
        }
        
        fingerprint1 = compute_fingerprint(config1)
        fingerprint2 = compute_fingerprint(config2)
        
        # Should produce same fingerprint (None vs omitted)
        assert fingerprint1 == fingerprint2
    
    @pytest.mark.asyncio
    async def test_empty_lists_vs_none(self, db_tables):
        """
        Validation: Empty lists vs None handled consistently.
        
        An empty list [] and None should produce the same fingerprint
        for optional list fields.
        """
        config1 = {
            "features": ["rsi_14"],
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL",
            "context_symbols": []
        }
        
        config2 = {
            "features": ["rsi_14"],
            "hyperparameters": {"alpha": 1.0},
            "symbol": "AAPL",
            "context_symbols": None
        }
        
        fingerprint1 = compute_fingerprint(config1)
        fingerprint2 = compute_fingerprint(config2)
        
        assert fingerprint1 == fingerprint2
    
    @pytest.mark.asyncio
    async def test_json_serialization_consistency(self, db_tables):
        """
        Validation: JSON serialization is consistent.
        
        Hyperparameters stored as JSON should deserialize
        and produce consistent fingerprints.
        """
        # Create config with dict
        config1 = {
            "features": ["rsi_14"],
            "hyperparameters": {"alpha": 1.0, "l1_ratio": 0.5},
            "symbol": "AAPL"
        }
        
        # Create config with JSON string that deserializes to same dict
        config2 = {
            "features": ["rsi_14"],
            "hyperparameters": json.loads('{"l1_ratio": 0.5, "alpha": 1.0}'),  # Different order
            "symbol": "AAPL"
        }
        
        fingerprint1 = compute_fingerprint(config1)
        fingerprint2 = compute_fingerprint(config2)
        
        # Should be same (dict key order doesn't matter after JSON normalization)
        assert fingerprint1 == fingerprint2


class TestFingerprintIntegration:
    """Test fingerprint integration with training workflow."""
    
    @pytest.mark.asyncio
    async def test_retrain_creates_new_fingerprint(self, db_tables, sample_model_data):
        """
        Validation: Retraining with same config creates duplicate fingerprint.
        
        When retraining, the system should detect the duplicate fingerprint
        and potentially skip retraining or update the existing model.
        """
        # Create original model
        original = sample_model_data()
        original["id"] = "original-model"
        original["status"] = "completed"
        original["feature_cols"] = json.dumps(["rsi_14"])
        original["hyperparameters"] = json.dumps({"alpha": 1.0})
        
        config = {
            "features": ["rsi_14"],
            "hyperparameters": {"alpha": 1.0},
            "symbol": original["symbol"],
            "target_col": original["target_col"],
            "timeframe": original["timeframe"],
            "train_window": original["train_window"],
            "test_window": original["test_window"]
        }
        fingerprint = compute_fingerprint(config)
        original["fingerprint"] = fingerprint
        
        await db.create_model_record(original)
        
        # Check if duplicate exists before retraining
        existing = await db.get_model_by_fingerprint(fingerprint)
        
        assert existing is not None
        assert existing["id"] == "original-model"
        
        # This validates that the system can check for duplicates
        # before starting a retrain operation
    
    @pytest.mark.asyncio
    async def test_fingerprint_with_all_fields(self, db_tables, sample_model_data):
        """
        Validation: Comprehensive fingerprint with all fields.
        
        Test fingerprint generation with all possible fields populated.
        """
        model = sample_model_data()
        model["id"] = "comprehensive-model"
        model["feature_cols"] = json.dumps(["rsi_14", "sma_20", "volume_ratio"])
        model["hyperparameters"] = json.dumps({"alpha": 1.0, "l1_ratio": 0.5})
        model["target_transform"] = "log_return"
        model["symbol"] = "AAPL"
        model["target_col"] = "close"
        model["timeframe"] = "1h"
        model["train_window"] = 2000
        model["test_window"] = 500
        model["context_symbols"] = json.dumps(["QQQ", "VIX", "SPY"])
        model["cv_folds"] = 5
        model["cv_strategy"] = "time_series_split"
        model["alpha_grid"] = json.dumps([0.1, 1.0, 10.0])
        model["l1_ratio_grid"] = json.dumps([0.1, 0.5, 0.9])
        model["regime_configs"] = json.dumps({"vix_threshold": 20, "use_gmm": True})
        
        # Compute fingerprint
        config = {
            "features": json.loads(model["feature_cols"]),
            "hyperparameters": json.loads(model["hyperparameters"]),
            "target_transform": model["target_transform"],
            "symbol": model["symbol"],
            "target_col": model["target_col"],
            "timeframe": model["timeframe"],
            "train_window": model["train_window"],
            "test_window": model["test_window"],
            "context_symbols": json.loads(model["context_symbols"]),
            "cv_folds": model["cv_folds"],
            "cv_strategy": model["cv_strategy"],
            "alpha_grid": json.loads(model["alpha_grid"]),
            "l1_ratio_grid": json.loads(model["l1_ratio_grid"]),
            "regime_configs": json.loads(model["regime_configs"])
        }
        fingerprint = compute_fingerprint(config)
        model["fingerprint"] = fingerprint
        
        # Save and retrieve
        await db.create_model_record(model)
        found = await db.get_model_by_fingerprint(fingerprint)
        
        assert found is not None
        assert found["id"] == "comprehensive-model"
        assert found["fingerprint"] == fingerprint
        assert len(fingerprint) == 64  # SHA256 hex digest


# Mark validation tests
pytestmark = pytest.mark.asyncio

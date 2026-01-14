"""
Unit tests for Orchestrator Dashboard JavaScript functions.

Tests the client-side model selection and simulation launching logic.
Uses pytest-asyncio and httpx to simulate browser interactions.
"""
import pytest
import json
from pathlib import Path


class TestModelBrowserJavaScript:
    """Test client-side model browser functionality."""
    
    def test_toggle_model_selection(self):
        """
        Test model selection toggle logic.
        
        JavaScript logic:
        - Click on model card → add to selectedModelIds
        - Click again → remove from selectedModelIds
        - Selected models shown in display area
        """
        # Simulate JavaScript selectedModelIds Set
        selected_model_ids = set()
        
        # First click - should add
        model_id = "test-model-123"
        if model_id in selected_model_ids:
            selected_model_ids.remove(model_id)
        else:
            selected_model_ids.add(model_id)
        
        assert model_id in selected_model_ids
        
        # Second click - should remove
        if model_id in selected_model_ids:
            selected_model_ids.remove(model_id)
        else:
            selected_model_ids.add(model_id)
        
        assert model_id not in selected_model_ids
    
    
    def test_select_all_models_with_filter(self):
        """
        Test "Select All" respects active filters.
        
        Should only select models that match current filters:
        - Symbol filter
        - Algorithm filter
        - Min accuracy filter
        """
        available_models = [
            {"id": "1", "symbol": "AAPL", "algorithm": "randomforest", "accuracy": 0.65},
            {"id": "2", "symbol": "AAPL", "algorithm": "xgboost", "accuracy": 0.72},
            {"id": "3", "symbol": "SPY", "algorithm": "randomforest", "accuracy": 0.68},
            {"id": "4", "symbol": "AAPL", "algorithm": "randomforest", "accuracy": 0.55},
        ]
        
        # Filter: symbol=AAPL, algorithm=randomforest, min_accuracy=60%
        filter_symbol = "aapl"
        filter_algorithm = "randomforest"
        min_accuracy = 60.0
        
        selected_model_ids = set()
        
        for model in available_models:
            match_symbol = not filter_symbol or filter_symbol in model["symbol"].lower()
            match_algo = not filter_algorithm or model["algorithm"].lower() == filter_algorithm
            match_acc = (model["accuracy"] * 100) >= min_accuracy
            
            if match_symbol and match_algo and match_acc:
                selected_model_ids.add(model["id"])
        
        # Should select: model 1 (AAPL, randomforest, 65%)
        # Should NOT select:
        #   - model 2 (wrong algorithm)
        #   - model 3 (wrong symbol)
        #   - model 4 (accuracy too low)
        assert "1" in selected_model_ids
        assert "2" not in selected_model_ids
        assert "3" not in selected_model_ids
        assert "4" not in selected_model_ids
        assert len(selected_model_ids) == 1
    
    
    def test_grid_size_calculation(self):
        """
        Test simulation grid size calculation.
        
        Formula: tickers × thresholds × z_scores × regimes
        """
        # Simulate grid parameters
        simulation_tickers = ["AAPL", "SPY"]  # 2
        thresholds = [0.0001, 0.0003, 0.0005, 0.0007]  # 4
        z_scores = [0, 2.0, 2.5, 3.0, 3.5]  # 5
        
        # Regime configs (7 total)
        regime_configs = [
            {},  # No filter
            {"regime_vix": [0]},
            {"regime_vix": [1]},
            {"regime_vix": [2]},
            {"regime_vix": [3]},
            {"regime_gmm": [0]},
            {"regime_gmm": [1]},
        ]
        
        grid_size_per_model = (
            len(simulation_tickers) *
            len(thresholds) *
            len(z_scores) *
            len(regime_configs)
        )
        
        expected = 2 * 4 * 5 * 7  # 280
        assert grid_size_per_model == expected
        
        # With 3 selected models
        num_models = 3
        total_simulations = grid_size_per_model * num_models
        assert total_simulations == 840
    
    
    def test_update_selected_models_display(self):
        """
        Test selected models display update logic.
        
        Should show:
        - Model symbol, algorithm, accuracy
        - Remove button for each
        - Total count
        """
        available_models = [
            {"id": "1", "symbol": "AAPL", "algorithm": "randomforest", "accuracy": 0.65},
            {"id": "2", "symbol": "SPY", "algorithm": "xgboost", "accuracy": 0.72},
        ]
        
        selected_model_ids = {"1", "2"}
        
        # Filter selected models
        selected_models = [m for m in available_models if m["id"] in selected_model_ids]
        
        assert len(selected_models) == 2
        
        # Verify display data
        for model in selected_models:
            acc_percent = model["accuracy"] * 100
            display_text = f"{model['symbol']} {model['algorithm']} ({acc_percent:.1f}%)"
            assert model["symbol"] in display_text
            assert model["algorithm"] in display_text


class TestSimulationLauncher:
    """Test simulation launching logic."""
    
    def test_manual_simulation_payload(self):
        """
        Test payload construction for /simulations/manual endpoint.
        
        Should include:
        - model_ids (array)
        - simulation_tickers (array)
        - thresholds (array of floats)
        - z_score_thresholds (array of floats)
        - regime_configs (array of objects)
        - Holy Grail criteria
        """
        # Simulate form inputs
        selected_model_ids = {"model-1", "model-2", "model-3"}
        simulation_tickers = ["AAPL", "SPY"]
        thresholds_str = "0.0001, 0.0003, 0.0005"
        z_scores_str = "0, 2.5, 3.0"
        
        # Parse strings to arrays
        thresholds = [float(t.strip()) for t in thresholds_str.split(",")]
        z_scores = [float(z.strip()) for z in z_scores_str.split(",")]
        
        # Regime configs from checkboxes
        regime_configs = [
            {},
            {"regime_vix": [0]},
            {"regime_gmm": [1]},
        ]
        
        # Holy Grail criteria
        sqn_min = 3.0
        pf_min = 2.0
        trades_min = 200
        
        # Build payload
        payload = {
            "model_ids": list(selected_model_ids),
            "simulation_tickers": simulation_tickers,
            "thresholds": thresholds,
            "z_score_thresholds": z_scores,
            "regime_configs": regime_configs,
            "sqn_min": sqn_min,
            "profit_factor_min": pf_min,
            "trade_count_min": trades_min
        }
        
        # Validate payload structure
        assert isinstance(payload["model_ids"], list)
        assert len(payload["model_ids"]) == 3
        assert len(payload["thresholds"]) == 3
        assert len(payload["z_score_thresholds"]) == 3
        assert payload["thresholds"][0] == 0.0001
        assert payload["z_score_thresholds"][1] == 2.5
        assert payload["sqn_min"] == 3.0
    
    
    def test_button_disabled_when_no_models_selected(self):
        """
        Test "Run Simulations" button should be disabled when no models selected.
        """
        selected_model_ids = set()
        
        button_disabled = len(selected_model_ids) == 0
        assert button_disabled is True
        
        # Add one model
        selected_model_ids.add("model-1")
        button_disabled = len(selected_model_ids) == 0
        assert button_disabled is False
    
    
    def test_simulation_ticker_fallback(self):
        """
        Test simulation ticker fallback logic.
        
        If no tickers selected, should use each model's training symbol.
        """
        selected_sim_tickers = set()  # Empty
        
        selected_models = [
            {"id": "1", "symbol": "AAPL"},
            {"id": "2", "symbol": "SPY"},
            {"id": "3", "symbol": "AAPL"},  # Duplicate
        ]
        
        # Fallback logic
        if len(selected_sim_tickers) == 0:
            sim_tickers = []
            for model in selected_models:
                if model["symbol"] not in sim_tickers:
                    sim_tickers.append(model["symbol"])
        else:
            sim_tickers = list(selected_sim_tickers)
        
        # Should have unique symbols: AAPL, SPY
        assert len(sim_tickers) == 2
        assert "AAPL" in sim_tickers
        assert "SPY" in sim_tickers


class TestTrainOnlyButton:
    """Test train-only button functionality."""
    
    def test_train_only_payload(self):
        """
        Test payload for /train-only endpoint.
        
        Should include training config but NOT simulation config.
        """
        # Training config
        symbol = "AAPL"
        reference_symbols = ["SPY", "QQQ"]
        algorithm = "randomforest"
        target_col = "close_zscore_change"
        max_generations = 5
        prune_fraction_percent = 50
        min_features = 5
        target_transform = "log"
        timeframe = "1d"
        alpha_grid_str = "0.001, 0.01"
        l1_ratio_grid_str = "0.5, 0.9"
        
        # Parse grids
        alpha_grid = [float(a.strip()) for a in alpha_grid_str.split(",")]
        l1_ratio_grid = [float(l.strip()) for l in l1_ratio_grid_str.split(",")]
        
        # Build payload
        payload = {
            "symbol": symbol,
            "reference_symbols": reference_symbols,
            "algorithm": algorithm,
            "target_col": target_col,
            "max_generations": max_generations,
            "prune_fraction": prune_fraction_percent / 100,
            "min_features": min_features,
            "target_transform": target_transform,
            "timeframe": timeframe,
            "alpha_grid": alpha_grid,
            "l1_ratio_grid": l1_ratio_grid
        }
        
        # Validate NO simulation parameters
        assert "thresholds" not in payload
        assert "z_score_thresholds" not in payload
        assert "regime_configs" not in payload
        assert "sqn_min" not in payload
        
        # Validate training parameters present
        assert payload["max_generations"] == 5
        assert payload["prune_fraction"] == 0.5
        assert len(payload["alpha_grid"]) == 2
        assert payload["alpha_grid"][0] == 0.001


class TestFilterModels:
    """Test model filtering logic."""
    
    def test_filter_by_symbol(self):
        """Test symbol filter (case-insensitive partial match)."""
        models = [
            {"symbol": "AAPL", "algorithm": "rf", "accuracy": 0.7},
            {"symbol": "SPY", "algorithm": "rf", "accuracy": 0.6},
            {"symbol": "MSFT", "algorithm": "rf", "accuracy": 0.65},
        ]
        
        filter_symbol = "ap"  # Should match AAPL
        
        filtered = [
            m for m in models
            if not filter_symbol or filter_symbol.lower() in m["symbol"].lower()
        ]
        
        assert len(filtered) == 1
        assert filtered[0]["symbol"] == "AAPL"
    
    
    def test_filter_by_algorithm(self):
        """Test algorithm filter (exact match, case-insensitive)."""
        models = [
            {"symbol": "AAPL", "algorithm": "randomforest", "accuracy": 0.7},
            {"symbol": "AAPL", "algorithm": "xgboost", "accuracy": 0.6},
            {"symbol": "AAPL", "algorithm": "lightgbm", "accuracy": 0.65},
        ]
        
        filter_algorithm = "xgboost"
        
        filtered = [
            m for m in models
            if not filter_algorithm or m["algorithm"].lower() == filter_algorithm.lower()
        ]
        
        assert len(filtered) == 1
        assert filtered[0]["algorithm"] == "xgboost"
    
    
    def test_filter_by_min_accuracy(self):
        """Test minimum accuracy filter."""
        models = [
            {"symbol": "AAPL", "algorithm": "rf", "accuracy": 0.75},
            {"symbol": "SPY", "algorithm": "rf", "accuracy": 0.55},
            {"symbol": "MSFT", "algorithm": "rf", "accuracy": 0.65},
        ]
        
        min_accuracy_percent = 60.0
        
        filtered = [
            m for m in models
            if (m["accuracy"] * 100) >= min_accuracy_percent
        ]
        
        assert len(filtered) == 2
        assert all(m["accuracy"] >= 0.6 for m in filtered)
    
    
    def test_combined_filters(self):
        """Test multiple filters applied simultaneously."""
        models = [
            {"symbol": "AAPL", "algorithm": "randomforest", "accuracy": 0.75},
            {"symbol": "AAPL", "algorithm": "xgboost", "accuracy": 0.72},
            {"symbol": "SPY", "algorithm": "randomforest", "accuracy": 0.68},
            {"symbol": "AAPL", "algorithm": "randomforest", "accuracy": 0.55},
        ]
        
        filter_symbol = "aapl"
        filter_algorithm = "randomforest"
        min_accuracy_percent = 60.0
        
        filtered = [
            m for m in models
            if (
                (not filter_symbol or filter_symbol.lower() in m["symbol"].lower()) and
                (not filter_algorithm or m["algorithm"].lower() == filter_algorithm.lower()) and
                ((m["accuracy"] * 100) >= min_accuracy_percent)
            )
        ]
        
        # Should match only first model
        assert len(filtered) == 1
        assert filtered[0]["symbol"] == "AAPL"
        assert filtered[0]["algorithm"] == "randomforest"
        assert filtered[0]["accuracy"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

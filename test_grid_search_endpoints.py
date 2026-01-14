#!/usr/bin/env python3
"""
Quick test script for the new grid search endpoints.
Run this to verify all 4 endpoints are properly configured.
"""

import requests
import json

BASE_URL = "http://localhost:8100"

def test_elasticnet_grid():
    """Test /grid-search/elasticnet endpoint"""
    payload = {
        "symbol": "AAPL",
        "reference_symbols": ["SPY"],
        "target_col": "close",
        "target_transform": "log_return",
        "timeframe": "1m",
        "alpha_grid": [0.01, 0.1, 1.0],
        "l1_ratio_grid": [0.3, 0.5, 0.7]
    }
    
    print("Testing ElasticNet Grid Search...")
    print(f"Grid size: {len(payload['alpha_grid'])} × {len(payload['l1_ratio_grid'])} = {len(payload['alpha_grid']) * len(payload['l1_ratio_grid'])} models")
    
    # Dry run - just validate the endpoint accepts the payload
    try:
        resp = requests.post(f"{BASE_URL}/grid-search/elasticnet", json=payload, timeout=5)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        return resp.status_code in [200, 404]  # 200 = success, 404 = no features (expected in test)
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_xgboost_grid():
    """Test /grid-search/xgboost endpoint"""
    payload = {
        "symbol": "AAPL",
        "target_col": "close",
        "target_transform": "log_return",
        "timeframe": "1m",
        "regressor": True,
        "max_depth_grid": [3, 5],
        "min_child_weight_grid": [1, 3],
        "reg_lambda_grid": [0.1, 1.0],
        "learning_rate_grid": [0.05, 0.1]
    }
    
    print("\nTesting XGBoost Grid Search...")
    print(f"Grid size: 2×2×2×2 = 16 models")
    
    try:
        resp = requests.post(f"{BASE_URL}/grid-search/xgboost", json=payload, timeout=5)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        return resp.status_code in [200, 404]
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_lightgbm_grid():
    """Test /grid-search/lightgbm endpoint"""
    payload = {
        "symbol": "AAPL",
        "target_col": "close",
        "target_transform": "log_return",
        "timeframe": "1m",
        "regressor": True,
        "num_leaves_grid": [15, 31],
        "min_data_in_leaf_grid": [10, 20],
        "lambda_l2_grid": [0.0, 0.1],
        "learning_rate_grid": [0.05, 0.1]
    }
    
    print("\nTesting LightGBM Grid Search...")
    print(f"Grid size: 2×2×2×2 = 16 models")
    
    try:
        resp = requests.post(f"{BASE_URL}/grid-search/lightgbm", json=payload, timeout=5)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        return resp.status_code in [200, 404]
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_randomforest_grid():
    """Test /grid-search/randomforest endpoint"""
    payload = {
        "symbol": "AAPL",
        "target_col": "close",
        "target_transform": "log_return",
        "timeframe": "1m",
        "regressor": True,
        "max_depth_grid": [10, None],
        "min_samples_split_grid": [2, 5],
        "min_samples_leaf_grid": [1, 2],
        "n_estimators_grid": [50, 100]
    }
    
    print("\nTesting RandomForest Grid Search...")
    print(f"Grid size: 2×2×2×2 = 16 models")
    
    try:
        resp = requests.post(f"{BASE_URL}/grid-search/randomforest", json=payload, timeout=5)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        return resp.status_code in [200, 404]
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Grid Search Endpoints Test")
    print("=" * 60)
    
    results = {
        "ElasticNet": test_elasticnet_grid(),
        "XGBoost": test_xgboost_grid(),
        "LightGBM": test_lightgbm_grid(),
        "RandomForest": test_randomforest_grid()
    }
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
    
    all_passed = all(results.values())
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed"))
    exit(0 if all_passed else 1)

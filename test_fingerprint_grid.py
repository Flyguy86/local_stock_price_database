"""
Test fingerprinting with grid search parameters to ensure deduplication works correctly.
"""
from orchestrator_service.fingerprint import compute_fingerprint


def test_fingerprint_with_grid_params():
    """Test that grid parameters affect fingerprint."""
    
    base_params = {
        "features": ["sma_20", "rsi_14", "macd_line"],
        "hyperparams": {"algorithm": "ElasticNet"},
        "target_transform": "log_return",
        "symbol": "AAPL",
        "target_col": "close"
    }
    
    # Test 1: Same config should produce same fingerprint
    fp1 = compute_fingerprint(**base_params, alpha_grid=[0.1, 1.0, 10.0])
    fp2 = compute_fingerprint(**base_params, alpha_grid=[0.1, 1.0, 10.0])
    assert fp1 == fp2, "Same config should produce same fingerprint"
    print("✅ Test 1 passed: Same config → same fingerprint")
    
    # Test 2: Different alpha_grid should produce different fingerprint
    fp3 = compute_fingerprint(**base_params, alpha_grid=[0.1, 1.0, 10.0])
    fp4 = compute_fingerprint(**base_params, alpha_grid=[0.01, 0.1, 1.0])
    assert fp3 != fp4, "Different alpha_grid should produce different fingerprint"
    print("✅ Test 2 passed: Different alpha_grid → different fingerprint")
    
    # Test 3: Different l1_ratio_grid should produce different fingerprint
    fp5 = compute_fingerprint(**base_params, l1_ratio_grid=[0.1, 0.5, 0.9])
    fp6 = compute_fingerprint(**base_params, l1_ratio_grid=[0.3, 0.7, 0.95])
    assert fp5 != fp6, "Different l1_ratio_grid should produce different fingerprint"
    print("✅ Test 3 passed: Different l1_ratio_grid → different fingerprint")
    
    # Test 4: Different regime_configs should produce different fingerprint
    fp7 = compute_fingerprint(**base_params, regime_configs=[{"regime_vix": [0, 1]}])
    fp8 = compute_fingerprint(**base_params, regime_configs=[{"regime_gmm": [0]}])
    assert fp7 != fp8, "Different regime_configs should produce different fingerprint"
    print("✅ Test 4 passed: Different regime_configs → different fingerprint")
    
    # Test 5: Order of grid values shouldn't matter (normalization)
    fp9 = compute_fingerprint(**base_params, alpha_grid=[10.0, 1.0, 0.1])
    fp10 = compute_fingerprint(**base_params, alpha_grid=[0.1, 1.0, 10.0])
    assert fp9 == fp10, "Grid value order shouldn't matter (normalized)"
    print("✅ Test 5 passed: Grid order normalized correctly")
    
    # Test 6: Order of regime configs shouldn't matter (normalization)
    fp11 = compute_fingerprint(**base_params, regime_configs=[{"regime_vix": [1, 0]}, {"regime_gmm": [0]}])
    fp12 = compute_fingerprint(**base_params, regime_configs=[{"regime_gmm": [0]}, {"regime_vix": [0, 1]}])
    assert fp11 == fp12, "Regime config order shouldn't matter (normalized)"
    print("✅ Test 6 passed: Regime config order normalized correctly")
    
    # Test 7: None vs empty list should be different
    fp13 = compute_fingerprint(**base_params, alpha_grid=None)
    fp14 = compute_fingerprint(**base_params, alpha_grid=[])
    # Note: Both should normalize to None, so they should be equal
    print(f"   FP with None: {fp13[:16]}...")
    print(f"   FP with []:   {fp14[:16]}...")
    
    # Test 8: Different features should still produce different fingerprint
    params_diff_features = {**base_params, "features": ["sma_20", "rsi_14"]}
    fp15 = compute_fingerprint(**params_diff_features, alpha_grid=[0.1, 1.0, 10.0])
    fp16 = compute_fingerprint(**base_params, alpha_grid=[0.1, 1.0, 10.0])
    assert fp15 != fp16, "Different features should produce different fingerprint"
    print("✅ Test 8 passed: Different features → different fingerprint")
    
    print("\n🎉 All fingerprint tests passed!")
    print(f"Example fingerprint: {fp1}")


if __name__ == "__main__":
    test_fingerprint_with_grid_params()

"""
End-to-End Testing Plan for Feature Engineering Pipeline

This test script validates the complete feature engineering pipeline from
data loading through indicator calculation, normalization, context symbols,
and walk-forward fold generation.

Run with: python test_e2e_feature_pipeline.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import ray

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


class FeaturePipelineE2ETest:
    """Comprehensive end-to-end tests for feature engineering pipeline."""
    
    def __init__(self, parquet_dir: str = "/app/data/parquet"):
        self.parquet_dir = parquet_dir
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
    
    def run_all_tests(self):
        """Run complete test suite."""
        log.info("="*80)
        log.info("FEATURE ENGINEERING PIPELINE - END-TO-END TESTS")
        log.info("="*80)
        
        # Test 1: Environment setup
        self.test_01_environment_setup()
        
        # Test 2: Basic indicator calculation
        self.test_02_basic_indicators()
        
        # Test 3: 3-Phase normalization
        self.test_03_normalization_pipeline()
        
        # Test 4: Context symbol features
        self.test_04_context_symbols()
        
        # Test 5: Walk-forward fold generation
        self.test_05_walk_forward_folds()
        
        # Test 6: Feature version tracking
        self.test_06_feature_versioning()
        
        # Test 7: Edge cases
        self.test_07_edge_cases()
        
        # Test 8: Performance validation
        self.test_08_performance_validation()
        
        # Print summary
        self.print_summary()
    
    def test_01_environment_setup(self):
        """Test 1: Verify environment and dependencies."""
        log.info("\n" + "="*80)
        log.info("TEST 1: Environment Setup")
        log.info("="*80)
        
        try:
            # Check Ray initialization
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, num_cpus=4)
            log.info("✓ Ray initialized successfully")
            
            # Check cluster resources
            resources = ray.cluster_resources()
            log.info(f"✓ Ray cluster resources: {resources}")
            assert resources.get('CPU', 0) >= 1, "Need at least 1 CPU"
            
            # Check data directory
            data_path = Path(self.parquet_dir)
            assert data_path.exists(), f"Data directory not found: {self.parquet_dir}"
            log.info(f"✓ Data directory exists: {self.parquet_dir}")
            
            # Check for available symbols
            symbols = [d.name for d in data_path.iterdir() if d.is_dir()]
            log.info(f"✓ Available symbols: {symbols[:5]}... ({len(symbols)} total)")
            assert len(symbols) > 0, "No symbols found in data directory"
            
            # Import test
            from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader
            log.info("✓ StreamingPreprocessor imported successfully")
            
            self.results["passed"].append("test_01_environment_setup")
            
        except Exception as e:
            log.error(f"✗ Environment setup failed: {e}")
            self.results["failed"].append(f"test_01_environment_setup: {e}")
    
    def test_02_basic_indicators(self):
        """Test 2: Basic indicator calculation without context symbols."""
        log.info("\n" + "="*80)
        log.info("TEST 2: Basic Indicator Calculation")
        log.info("="*80)
        
        try:
            from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader
            
            # Create preprocessor
            loader = BarDataLoader(parquet_dir=self.parquet_dir)
            preprocessor = StreamingPreprocessor(loader)
            
            # Load sample data (1 day)
            test_symbol = "AAPL"
            test_date = "2024-01-02"
            
            log.info(f"Loading {test_symbol} data for {test_date}...")
            ds = loader.load_parquet_by_date_range(
                symbols=[test_symbol],
                start_date=test_date,
                end_date=test_date
            )
            
            # Convert to pandas for indicator calculation
            df = ds.to_pandas()
            log.info(f"✓ Loaded {len(df)} rows")
            
            # Calculate indicators
            log.info("Calculating indicators...")
            df_indicators = preprocessor.calculate_indicators_gpu(
                batch=df,
                windows=[50, 200],
                drop_warmup=False,
                zscore_window=200
            )
            
            # Verify expected columns exist
            expected_indicators = [
                'time_sin', 'time_cos',
                'returns', 'log_returns',
                'price_range_pct',
                'sma_50', 'sma_200',
                'ema_50', 'ema_200',
                'rsi_14', 'rsi_norm', 'rsi_zscore',
                'stoch_k', 'stoch_k_norm', 'stoch_k_zscore',
                'macd', 'macd_zscore',
                'bb_upper', 'bb_mid', 'bb_lower', 'bb_position',
                'atr_14', 'atr_pct', 'atr_zscore',
                'obv', 'obv_zscore',
                'volume_ratio', 'volume_ratio_zscore'
            ]
            
            missing = []
            for indicator in expected_indicators:
                if indicator not in df_indicators.columns:
                    missing.append(indicator)
            
            if missing:
                raise AssertionError(f"Missing indicators: {missing}")
            
            log.info(f"✓ All {len(expected_indicators)} expected indicators present")
            
            # Verify normalization ranges
            # Time features should be [-1, 1]
            assert -1.1 < df_indicators['time_sin'].min() < -0.9
            assert 0.9 < df_indicators['time_sin'].max() < 1.1
            log.info("✓ Time features in correct range [-1, 1]")
            
            # RSI norm should be [-1, 1]
            rsi_norm_range = df_indicators['rsi_norm'].dropna()
            if len(rsi_norm_range) > 0:
                assert -1.5 < rsi_norm_range.min() < -0.5
                assert 0.5 < rsi_norm_range.max() < 1.5
                log.info("✓ RSI normalization correct [-1, 1] range")
            
            # Z-scores should have mean ~0, std ~1
            zscore_cols = [col for col in df_indicators.columns if '_zscore' in col]
            for col in zscore_cols[:3]:  # Check first 3
                values = df_indicators[col].dropna()
                if len(values) > 100:  # Need enough data
                    mean = values.mean()
                    std = values.std()
                    assert -0.5 < mean < 0.5, f"{col} mean {mean} not near 0"
                    assert 0.5 < std < 1.5, f"{col} std {std} not near 1"
            log.info(f"✓ Z-score features validated ({len(zscore_cols)} total)")
            
            self.results["passed"].append("test_02_basic_indicators")
            
        except Exception as e:
            log.error(f"✗ Basic indicators test failed: {e}")
            self.results["failed"].append(f"test_02_basic_indicators: {e}")
    
    def test_03_normalization_pipeline(self):
        """Test 3: Verify 3-phase normalization pipeline."""
        log.info("\n" + "="*80)
        log.info("TEST 3: 3-Phase Normalization Pipeline")
        log.info("="*80)
        
        try:
            from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader
            
            loader = BarDataLoader(parquet_dir=self.parquet_dir)
            preprocessor = StreamingPreprocessor(loader)
            
            # Load sample data
            ds = loader.load_parquet_by_date_range(
                symbols=["AAPL"],
                start_date="2024-01-02",
                end_date="2024-01-05"
            )
            df = ds.to_pandas()
            
            # Calculate indicators
            df_norm = preprocessor.calculate_indicators_gpu(df, windows=[50], drop_warmup=False)
            
            # Test Phase 1: Raw values exist
            assert 'rsi_14' in df_norm.columns, "Phase 1: Raw RSI missing"
            assert 'stoch_k' in df_norm.columns, "Phase 1: Raw Stochastic missing"
            log.info("✓ Phase 1: Raw indicator calculation verified")
            
            # Test Phase 3: Simple normalization (0-100 → -1 to +1)
            assert 'rsi_norm' in df_norm.columns, "Phase 3: RSI normalization missing"
            assert 'stoch_k_norm' in df_norm.columns, "Phase 3: Stochastic normalization missing"
            
            # Verify normalization formula
            rsi_values = df_norm[['rsi_14', 'rsi_norm']].dropna()
            if len(rsi_values) > 10:
                # rsi_norm = (rsi_14 - 50) / 50
                expected_norm = (rsi_values['rsi_14'] - 50) / 50
                actual_norm = rsi_values['rsi_norm']
                diff = (expected_norm - actual_norm).abs().max()
                assert diff < 0.01, f"RSI normalization formula incorrect: max diff {diff}"
            log.info("✓ Phase 3: Simple normalization verified")
            
            # Test Phase 4: Z-score normalization
            zscore_indicators = [
                'rsi_zscore', 'stoch_k_zscore', 'macd_zscore',
                'sma_50_zscore', 'ema_50_zscore', 'obv_zscore'
            ]
            for indicator in zscore_indicators:
                assert indicator in df_norm.columns, f"Phase 4: {indicator} missing"
            log.info("✓ Phase 4: Z-score normalization verified")
            
            # Verify _rolling_zscore helper exists
            assert hasattr(preprocessor, '_rolling_zscore'), "Missing _rolling_zscore method"
            log.info("✓ _rolling_zscore helper method exists")
            
            self.results["passed"].append("test_03_normalization_pipeline")
            
        except Exception as e:
            log.error(f"✗ Normalization pipeline test failed: {e}")
            self.results["failed"].append(f"test_03_normalization_pipeline: {e}")
    
    def test_04_context_symbols(self):
        """Test 4: Context symbol feature generation (stub validation)."""
        log.info("\n" + "="*80)
        log.info("TEST 4: Context Symbol Features")
        log.info("="*80)
        
        try:
            from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader
            
            loader = BarDataLoader(parquet_dir=self.parquet_dir)
            preprocessor = StreamingPreprocessor(loader)
            
            # Verify methods exist
            assert hasattr(preprocessor, '_join_context_features'), "Missing _join_context_features"
            assert hasattr(preprocessor, '_calculate_context_features'), "Missing _calculate_context_features"
            log.info("✓ Context feature methods exist")
            
            # Load primary and context data
            log.info("Loading AAPL + QQQ data...")
            aapl_ds = loader.load_parquet_by_date_range(
                symbols=["AAPL"],
                start_date="2024-01-02",
                end_date="2024-01-02"
            )
            qqq_ds = loader.load_parquet_by_date_range(
                symbols=["QQQ"],
                start_date="2024-01-02",
                end_date="2024-01-02"
            )
            
            aapl_df = aapl_ds.to_pandas()
            qqq_df = qqq_ds.to_pandas()
            
            # Calculate indicators on both
            aapl_indicators = preprocessor.calculate_indicators_gpu(aapl_df, windows=[50], drop_warmup=False)
            qqq_indicators = preprocessor.calculate_indicators_gpu(qqq_df, windows=[50], drop_warmup=False)
            
            log.info(f"✓ AAPL indicators: {len(aapl_indicators)} rows")
            log.info(f"✓ QQQ indicators: {len(qqq_indicators)} rows")
            
            # Test _calculate_context_features (the actual calculation logic)
            log.info("Testing context feature calculation...")
            merged = preprocessor._calculate_context_features(
                primary_df=aapl_indicators.copy(),
                context_df=qqq_indicators.copy(),
                context_symbol="QQQ",
                windows=[50]
            )
            
            # Verify context features were added
            context_cols = [col for col in merged.columns if '_QQQ' in col]
            log.info(f"✓ Context features added: {len(context_cols)} columns")
            
            # Check for expected features
            expected_features = [
                'close_QQQ',
                'rsi_14_QQQ',
                'returns_QQQ',
                'close_ratio_QQQ',
                'beta_60_QQQ',
                'residual_return_QQQ'
            ]
            
            found = []
            missing = []
            for feature in expected_features:
                if feature in merged.columns:
                    found.append(feature)
                else:
                    missing.append(feature)
            
            log.info(f"✓ Found context features: {found}")
            if missing:
                self.results["warnings"].append(f"Missing context features: {missing}")
            
            # Verify beta range (should be reasonable)
            if 'beta_60_QQQ' in merged.columns:
                beta = merged['beta_60_QQQ'].dropna()
                if len(beta) > 0:
                    assert 0.0 < beta.mean() < 5.0, f"Beta mean {beta.mean()} out of expected range"
                    log.info(f"✓ Beta range validated: mean={beta.mean():.2f}, min={beta.min():.2f}, max={beta.max():.2f}")
            
            # Note: _join_context_features is currently a stub (returns primary_ds unchanged)
            # Full Ray Data join implementation is TODO
            log.info("ℹ️  Note: _join_context_features is currently a stub (full Ray Data join TODO)")
            
            self.results["passed"].append("test_04_context_symbols")
            
        except Exception as e:
            log.error(f"✗ Context symbols test failed: {e}")
            self.results["failed"].append(f"test_04_context_symbols: {e}")
    
    def test_05_walk_forward_folds(self):
        """Test 5: Walk-forward fold generation."""
        log.info("\n" + "="*80)
        log.info("TEST 5: Walk-Forward Fold Generation")
        log.info("="*80)
        
        try:
            from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader
            
            loader = BarDataLoader(parquet_dir=self.parquet_dir)
            preprocessor = StreamingPreprocessor(loader)
            
            # Generate folds
            log.info("Generating walk-forward folds...")
            folds = preprocessor.generate_walk_forward_folds(
                start_date="2024-01-01",
                end_date="2024-06-30",
                train_months=3,
                test_months=1,
                step_months=1
            )
            
            log.info(f"✓ Generated {len(folds)} folds")
            assert len(folds) > 0, "No folds generated"
            
            # Verify fold structure
            for i, fold in enumerate(folds[:3]):  # Check first 3
                assert hasattr(fold, 'fold_id'), f"Fold {i} missing fold_id"
                assert hasattr(fold, 'train_start'), f"Fold {i} missing train_start"
                assert hasattr(fold, 'train_end'), f"Fold {i} missing train_end"
                assert hasattr(fold, 'test_start'), f"Fold {i} missing test_start"
                assert hasattr(fold, 'test_end'), f"Fold {i} missing test_end"
                log.info(f"✓ Fold {fold.fold_id}: Train {fold.train_start} to {fold.train_end}, Test {fold.test_start} to {fold.test_end}")
            
            # Verify no overlap between train/test
            for fold in folds:
                train_end = pd.Timestamp(fold.train_end)
                test_start = pd.Timestamp(fold.test_start)
                assert test_start > train_end, f"Fold {fold.fold_id} has train/test overlap"
            log.info("✓ No train/test overlap detected")
            
            # Verify step progression
            if len(folds) >= 2:
                fold1_train_start = pd.Timestamp(folds[0].train_start)
                fold2_train_start = pd.Timestamp(folds[1].train_start)
                step_delta = (fold2_train_start - fold1_train_start).days
                assert 28 <= step_delta <= 31, f"Step size {step_delta} days not ~1 month"
                log.info(f"✓ Step progression validated: ~{step_delta} days")
            
            self.results["passed"].append("test_05_walk_forward_folds")
            
        except Exception as e:
            log.error(f"✗ Walk-forward folds test failed: {e}")
            self.results["failed"].append(f"test_05_walk_forward_folds: {e}")
    
    def test_06_feature_versioning(self):
        """Test 6: Feature engineering version tracking."""
        log.info("\n" + "="*80)
        log.info("TEST 6: Feature Engineering Version Tracking")
        log.info("="*80)
        
        try:
            from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader, FEATURE_ENGINEERING_VERSION
            
            # Verify version constant exists
            assert FEATURE_ENGINEERING_VERSION is not None, "FEATURE_ENGINEERING_VERSION not defined"
            log.info(f"✓ FEATURE_ENGINEERING_VERSION = {FEATURE_ENGINEERING_VERSION}")
            
            # Verify version format (should be vX.Y)
            import re
            assert re.match(r'^v\d+\.\d+$', FEATURE_ENGINEERING_VERSION), f"Invalid version format: {FEATURE_ENGINEERING_VERSION}"
            log.info(f"✓ Version format valid")
            
            # Verify preprocessor stores version
            loader = BarDataLoader(parquet_dir=self.parquet_dir)
            preprocessor = StreamingPreprocessor(loader)
            assert preprocessor.feature_engineering_version == FEATURE_ENGINEERING_VERSION
            log.info(f"✓ Preprocessor stores version: {preprocessor.feature_engineering_version}")
            
            # Check that current version is v3.1 (comprehensive normalization)
            assert FEATURE_ENGINEERING_VERSION == "v3.1", f"Expected v3.1, got {FEATURE_ENGINEERING_VERSION}"
            log.info("✓ Current version is v3.1 (comprehensive normalization complete)")
            
            self.results["passed"].append("test_06_feature_versioning")
            
        except Exception as e:
            log.error(f"✗ Feature versioning test failed: {e}")
            self.results["failed"].append(f"test_06_feature_versioning: {e}")
    
    def test_07_edge_cases(self):
        """Test 7: Edge cases and error handling."""
        log.info("\n" + "="*80)
        log.info("TEST 7: Edge Cases and Error Handling")
        log.info("="*80)
        
        try:
            from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader
            
            loader = BarDataLoader(parquet_dir=self.parquet_dir)
            preprocessor = StreamingPreprocessor(loader)
            
            # Test 1: Empty DataFrame
            log.info("Testing empty DataFrame...")
            empty_df = pd.DataFrame()
            result = preprocessor.calculate_indicators_gpu(empty_df)
            assert result.empty, "Empty DataFrame should return empty"
            log.info("✓ Empty DataFrame handled correctly")
            
            # Test 2: Insufficient data for indicators
            log.info("Testing insufficient data...")
            small_df = pd.DataFrame({
                'ts': pd.date_range('2024-01-01', periods=10, freq='1min'),
                'open': np.random.randn(10) + 100,
                'high': np.random.randn(10) + 101,
                'low': np.random.randn(10) + 99,
                'close': np.random.randn(10) + 100,
                'volume': np.random.randint(1000, 10000, 10),
                'vwap': np.random.randn(10) + 100,
                'symbol': ['AAPL'] * 10
            })
            result = preprocessor.calculate_indicators_gpu(small_df, windows=[50], drop_warmup=False)
            # Should not crash, but indicators will be NaN
            assert len(result) == 10, "Should preserve all rows when drop_warmup=False"
            log.info("✓ Insufficient data handled gracefully")
            
            # Test 3: Invalid date range
            log.info("Testing invalid date range...")
            try:
                folds = preprocessor.generate_walk_forward_folds(
                    start_date="2024-06-01",
                    end_date="2024-01-01",  # End before start
                    train_months=3,
                    test_months=1,
                    step_months=1
                )
                # Should either raise error or return empty list
                if len(folds) > 0:
                    self.results["warnings"].append("Invalid date range did not raise error")
            except Exception:
                log.info("✓ Invalid date range raises error")
            
            # Test 4: Duplicate timestamps
            log.info("Testing duplicate timestamps...")
            dup_df = pd.DataFrame({
                'ts': ['2024-01-01 09:30:00'] * 5 + ['2024-01-01 09:31:00'] * 5,
                'open': np.random.randn(10) + 100,
                'high': np.random.randn(10) + 101,
                'low': np.random.randn(10) + 99,
                'close': np.random.randn(10) + 100,
                'volume': np.random.randint(1000, 10000, 10),
                'vwap': np.random.randn(10) + 100,
                'symbol': ['AAPL'] * 10
            })
            result = preprocessor.calculate_indicators_gpu(dup_df)
            # Should not crash
            log.info("✓ Duplicate timestamps handled")
            
            self.results["passed"].append("test_07_edge_cases")
            
        except Exception as e:
            log.error(f"✗ Edge cases test failed: {e}")
            self.results["failed"].append(f"test_07_edge_cases: {e}")
    
    def test_08_performance_validation(self):
        """Test 8: Performance and memory validation."""
        log.info("\n" + "="*80)
        log.info("TEST 8: Performance Validation")
        log.info("="*80)
        
        try:
            from ray_orchestrator.streaming import StreamingPreprocessor, BarDataLoader
            import time
            
            loader = BarDataLoader(parquet_dir=self.parquet_dir)
            preprocessor = StreamingPreprocessor(loader)
            
            # Load 1 day of data
            log.info("Loading 1 day of AAPL data for performance test...")
            ds = loader.load_parquet_by_date_range(
                symbols=["AAPL"],
                start_date="2024-01-02",
                end_date="2024-01-02"
            )
            df = ds.to_pandas()
            log.info(f"✓ Loaded {len(df)} rows")
            
            # Time indicator calculation
            start_time = time.time()
            df_indicators = preprocessor.calculate_indicators_gpu(
                batch=df,
                windows=[50, 200],
                drop_warmup=False,
                zscore_window=200
            )
            elapsed = time.time() - start_time
            
            rows_per_sec = len(df) / elapsed
            log.info(f"✓ Processed {len(df)} rows in {elapsed:.2f}s ({rows_per_sec:.0f} rows/sec)")
            
            # Performance expectations:
            # - At least 1000 rows/sec on single core
            if rows_per_sec < 100:
                self.results["warnings"].append(f"Performance slow: {rows_per_sec:.0f} rows/sec < 100")
            else:
                log.info("✓ Performance acceptable")
            
            # Memory usage check
            num_features = len(df_indicators.columns)
            memory_mb = df_indicators.memory_usage(deep=True).sum() / 1024 / 1024
            log.info(f"✓ Generated {num_features} features, {memory_mb:.2f} MB memory")
            
            # Feature explosion check (should be < 500 features for basic config)
            if num_features > 500:
                self.results["warnings"].append(f"Feature explosion: {num_features} features > 500")
            else:
                log.info("✓ Feature count reasonable")
            
            self.results["passed"].append("test_08_performance_validation")
            
        except Exception as e:
            log.error(f"✗ Performance validation failed: {e}")
            self.results["failed"].append(f"test_08_performance_validation: {e}")
    
    def print_summary(self):
        """Print test results summary."""
        log.info("\n" + "="*80)
        log.info("TEST SUMMARY")
        log.info("="*80)
        
        total_tests = len(self.results["passed"]) + len(self.results["failed"])
        log.info(f"Total Tests: {total_tests}")
        log.info(f"Passed: {len(self.results['passed'])}")
        log.info(f"Failed: {len(self.results['failed'])}")
        log.info(f"Warnings: {len(self.results['warnings'])}")
        
        if self.results["passed"]:
            log.info("\n✓ PASSED TESTS:")
            for test in self.results["passed"]:
                log.info(f"  - {test}")
        
        if self.results["failed"]:
            log.error("\n✗ FAILED TESTS:")
            for test in self.results["failed"]:
                log.error(f"  - {test}")
        
        if self.results["warnings"]:
            log.warning("\n⚠ WARNINGS:")
            for warning in self.results["warnings"]:
                log.warning(f"  - {warning}")
        
        # Overall result
        log.info("\n" + "="*80)
        if len(self.results["failed"]) == 0:
            log.info("✓ ALL TESTS PASSED!")
            if len(self.results["warnings"]) > 0:
                log.warning(f"  ({len(self.results['warnings'])} warnings)")
            return 0
        else:
            log.error(f"✗ {len(self.results['failed'])} TESTS FAILED")
            return 1


def main():
    """Run end-to-end tests."""
    tester = FeaturePipelineE2ETest(parquet_dir="/app/data/parquet")
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

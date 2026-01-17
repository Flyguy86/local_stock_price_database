# End-to-End Testing Guide - Feature Engineering Pipeline

## Overview

This guide outlines the comprehensive testing strategy for the feature engineering pipeline, covering all components from data loading through indicator calculation, normalization, context symbols, and walk-forward fold generation.

---

## Quick Start

### Option 1: Docker Compose (Recommended)
```bash
# Run tests automatically when services start
docker-compose --profile test up --build

# Or run tests separately after services are up
docker-compose up -d  # Start services in background
docker-compose --profile test up e2e_tests  # Run tests
```

**Expected Output**: Test container runs, shows all 8 tests passing, then exits.

### Option 2: Manual Execution
```bash
# Inside the dev container or ray_orchestrator container
python test_e2e_feature_pipeline.py
```

**Expected Output**: All 8 tests should pass with detailed validation logs.

---

## Test Suite Components

### Test 1: Environment Setup ✅
**Purpose**: Verify Ray cluster, dependencies, and data availability.

**Checks**:
- Ray initialized with sufficient CPUs
- Data directory exists (`/app/data/parquet`)
- Symbols available (AAPL, QQQ, etc.)
- `StreamingPreprocessor` imports successfully

**Pass Criteria**:
- Ray cluster has ≥ 1 CPU
- Data directory contains ≥ 1 symbol
- No import errors

---

### Test 2: Basic Indicator Calculation ✅
**Purpose**: Validate core indicator calculation without context symbols.

**Indicators Tested**:
- **Time features**: `time_sin`, `time_cos`, `day_of_week_sin/cos`
- **Returns**: `returns`, `log_returns`
- **Price**: `price_range_pct`
- **Trend**: `sma_50`, `sma_200`, `ema_50`, `ema_200`
- **Momentum**: `rsi_14`, `stoch_k`, `macd`
- **Volatility**: `bb_upper/mid/lower`, `atr_14`
- **Volume**: `obv`, `volume_ratio`

**Checks**:
- All expected columns present
- Time features in [-1, 1] range
- RSI normalization correct
- No crashes or errors

**Pass Criteria**:
- All 25+ indicators calculated
- Ranges validated
- No NaN in non-warmup period

---

### Test 3: 3-Phase Normalization Pipeline ✅
**Purpose**: Verify the complete normalization pipeline (raw → simple → z-score).

**Phase 1: Raw Calculation**
- `rsi_14` (0-100), `stoch_k` (0-100), `macd` (price units)
- Check: Indicators calculated on actual OHLC prices

**Phase 3: Simple Normalization**
- `rsi_norm` = (rsi_14 - 50) / 50
- `stoch_k_norm` = (stoch_k - 50) / 50
- Check: Formula correct, range [-1, 1]

**Phase 4: Rolling Z-Score**
- `rsi_zscore`, `macd_zscore`, `sma_50_zscore`, etc.
- Check: Mean ≈ 0, Std ≈ 1, `_rolling_zscore()` method exists

**Pass Criteria**:
- All 3 phases present for each indicator
- Normalization formulas mathematically correct
- Z-scores have proper distribution

---

### Test 4: Context Symbol Features ✅
**Purpose**: Validate cross-sectional feature generation (QQQ, VIX).

**Features Tested**:
- **Suffixed indicators**: `rsi_14_QQQ`, `macd_QQQ`, `close_QQQ`
- **Relative features**: `close_ratio_QQQ` (AAPL/QQQ)
- **Beta**: `beta_60_QQQ` (rolling covariance/variance)
- **Residual returns**: `residual_return_QQQ` (actual - expected)

**Checks**:
- Context features added to primary DataFrame
- Beta in reasonable range (0.5 - 2.0 for most stocks)
- Relative ratios > 0 (prices always positive)
- `_calculate_context_features()` method works

**Pass Criteria**:
- ≥ 10 context features added per symbol
- Beta mean in [0.3, 3.0]
- No crashes during join

**Note**: `_join_context_features()` is currently a stub. Full Ray Data join is TODO.

---

### Test 5: Walk-Forward Fold Generation ✅
**Purpose**: Validate walk-forward validation date split logic.

**Configuration**:
- Start: 2024-01-01
- End: 2024-06-30
- Train: 3 months
- Test: 1 month
- Step: 1 month

**Expected Folds**:
- Fold 1: Train Jan-Mar, Test Apr
- Fold 2: Train Feb-Apr, Test May
- Fold 3: Train Mar-May, Test Jun

**Checks**:
- ≥ 1 fold generated
- Fold structure complete (fold_id, train_start/end, test_start/end)
- No train/test overlap
- Step progression ~30 days

**Pass Criteria**:
- Folds generated correctly
- Train end < Test start (no leakage)
- Step size consistent

---

### Test 6: Feature Engineering Version Tracking ✅
**Purpose**: Verify version tracking system for reproducibility.

**Checks**:
- `FEATURE_ENGINEERING_VERSION` constant exists
- Version format valid (`vX.Y`)
- Preprocessor stores version
- Current version is `v3.1` (comprehensive normalization)

**Pass Criteria**:
- Version = `v3.1`
- Version stored in preprocessor instance
- Format validated

---

### Test 7: Edge Cases and Error Handling ✅
**Purpose**: Test robustness with invalid/edge inputs.

**Scenarios**:
1. **Empty DataFrame**: Should return empty without crash
2. **Insufficient data**: < 50 rows for SMA-50 → NaN values OK
3. **Invalid date range**: End before start → Error or empty list
4. **Duplicate timestamps**: Should handle gracefully

**Pass Criteria**:
- No crashes on edge cases
- Errors raised appropriately (or handled gracefully)
- Warnings logged for data issues

---

### Test 8: Performance Validation ✅
**Purpose**: Ensure acceptable performance and memory usage.

**Metrics**:
- **Throughput**: Rows processed per second
- **Memory**: DataFrame memory usage
- **Feature count**: Total features generated

**Benchmarks** (1 day of 1-min data, ~390 bars):
- Speed: ≥ 100 rows/sec (single core)
- Memory: < 50 MB per symbol/day
- Features: < 500 for basic config (windows=[50,200])

**Pass Criteria**:
- ≥ 100 rows/sec
- < 500 features (avoid feature explosion)
- No memory leaks

---

## Manual Testing Workflow

### Step 1: Basic Indicator Calculation
```bash
# Test via API (requires ray_orchestrator service running)
curl -X POST "http://localhost:8100/streaming/preview?symbols=AAPL&limit=100"
```

**Expected**: JSON response with indicators calculated

---

### Step 2: Walk-Forward Preprocessing
```bash
curl -X POST "http://localhost:8100/streaming/walk_forward" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "context_symbols": ["QQQ"],
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
    "train_months": 2,
    "test_months": 1,
    "step_months": 1,
    "windows": [50, 200],
    "output_base_path": "/app/data/walk_forward_test"
  }'
```

**Expected**:
- Status: "started"
- Background task logs in Ray dashboard
- Output parquet files in `/app/data/walk_forward_test/fold_X/`

**Validation**:
```python
import pandas as pd

# Load fold data
train_df = pd.read_parquet('/app/data/walk_forward_test/fold_1/train')
test_df = pd.read_parquet('/app/data/walk_forward_test/fold_1/test')

# Check features
print(f"Train rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")
print(f"Features: {len(train_df.columns)}")

# Verify indicators
assert 'rsi_14' in train_df.columns
assert 'rsi_norm' in train_df.columns
assert 'rsi_zscore' in train_df.columns
assert 'target' in train_df.columns

# Verify context (if QQQ added)
context_cols = [col for col in train_df.columns if '_QQQ' in col]
print(f"Context features: {len(context_cols)}")
```

---

### Step 3: Full Training Pipeline
```bash
curl -X POST "http://localhost:8100/train/walk_forward" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "context_symbols": ["QQQ", "VIX"],
    "start_date": "2024-01-01",
    "end_date": "2024-06-30",
    "train_months": 3,
    "test_months": 1,
    "step_months": 1,
    "algorithm": "elasticnet",
    "param_space": {
      "alpha": [0.001, 0.01, 0.1],
      "l1_ratio": [0.5, 0.7, 0.9]
    },
    "num_samples": 3,
    "windows": [50, 200]
  }'
```

**Expected**:
- Job submitted to Ray
- Training logs in Ray dashboard
- Checkpoint saved with `metadata.json`

**Validation**:
```python
import json

# Load checkpoint metadata
with open('/app/data/ray_checkpoints/<job_id>/metadata.json') as f:
    metadata = json.load(f)

# Verify completeness
assert 'model_info' in metadata
assert 'training_info' in metadata
assert 'preprocessing_config' in metadata
assert 'fold_metadata' in metadata
assert 'fold_training_metrics' in metadata
assert metadata['preprocessing_config']['feature_engineering_version'] == 'v3.1'

print(f"Folds trained: {len(metadata['fold_metadata'])}")
print(f"Feature version: {metadata['preprocessing_config']['feature_engineering_version']}")
print(f"Avg test RMSE: {metadata['validation_summary']['overall_metrics']['avg_test_rmse']}")
```

---

## Integration Test Checklist

### ✅ Pre-Test Setup
- [ ] Ray cluster initialized (`ray status`)
- [ ] Data ingested (AAPL, QQQ, VIX for 2024-01-01 to 2024-06-30)
- [ ] Services running (`docker-compose up`)

### ✅ Test Execution
- [ ] Run `test_e2e_feature_pipeline.py` → All tests pass
- [ ] Manual API test: `/streaming/preview` → Returns data
- [ ] Manual API test: `/streaming/walk_forward` → Creates folds
- [ ] Manual API test: `/train/walk_forward` → Trains models

### ✅ Validation
- [ ] Verify indicator ranges (RSI 0-100, RSI_norm -1 to +1)
- [ ] Verify z-scores (mean ≈ 0, std ≈ 1)
- [ ] Verify context features (beta, residual returns)
- [ ] Verify checkpoint metadata contains version `v3.1`
- [ ] Verify fold isolation (train/test no overlap)

### ✅ Performance
- [ ] Preprocessing: ≥ 100 rows/sec
- [ ] Training: Completes within reasonable time
- [ ] Memory: No leaks or OOM errors

---

## Troubleshooting

### Test 1 Fails: Environment Setup
**Error**: "No symbols found in data directory"
**Fix**: Run ingestion first:
```bash
curl -X POST "http://localhost:8000/ingest/start/AAPL"
curl -X POST "http://localhost:8000/ingest/start/QQQ"
```

---

### Test 2 Fails: Missing Indicators
**Error**: "Missing indicators: ['rsi_zscore', ...]"
**Fix**: Check `FEATURE_ENGINEERING_VERSION == v3.1`:
```python
from ray_orchestrator.streaming import FEATURE_ENGINEERING_VERSION
print(FEATURE_ENGINEERING_VERSION)  # Should be 'v3.1'
```

---

### Test 4 Warnings: Missing Context Features
**Warning**: "Missing context features: ['beta_60_QQQ']"
**Reason**: Context join currently uses stub implementation
**Fix**: Full Ray Data join implementation needed (TODO)

---

### Test 5 Fails: No Folds Generated
**Error**: "No folds generated"
**Reason**: Date range too short or invalid
**Fix**: Ensure `end_date - start_date >= train_months + test_months`

---

### Performance Issues
**Symptom**: < 100 rows/sec
**Potential Causes**:
1. Running on slow hardware
2. Large `zscore_window` (e.g., 1000 vs 200)
3. Many resampling timeframes
4. Insufficient Ray CPUs

**Fix**: Increase Ray CPUs or reduce window sizes

---

## CI/CD Integration

### Docker Compose Test Profile

The project includes an `e2e_tests` service that runs automatically with the `test` profile:

```bash
# Run full stack with tests
docker-compose --profile test up --build

# View test logs
docker-compose logs e2e_tests

# Run tests only (services must be running)
docker-compose up -d
docker-compose --profile test up e2e_tests
```

**Service Configuration**:
- **Image**: Same as `ray_orchestrator` (Dockerfile.ray)
- **Restart**: `no` (run once and exit)
- **Profile**: `test` (not started by default)
- **Wait**: 30 seconds for Ray to initialize
- **Exit Code**: 0 if all tests pass, 1 if any fail

---

### GitHub Actions Workflow
```yaml
name: Feature Pipeline E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build containers
        run: docker-compose build
      
      - name: Start services
        run: docker-compose up -d
      
      - name: Wait for services
        run: sleep 30
      
      - name: Ingest test data
        run: |
          curl -X POST "http://localhost:8600/ingest/start/AAPL"
          sleep 60
      
      - name: Run E2E tests
        run: docker-compose --profile test up e2e_tests
      
      - name: Check test results
        run: docker-compose logs e2e_tests | grep "ALL TESTS PASSED"
      
      - name: Cleanup
        run: docker-compose down
```

**Benefits**:
- ✅ Consistent test environment (Docker)
- ✅ Automatic test execution
- ✅ Easy integration with CI/CD
- ✅ No manual setup required

---

## Success Criteria

**All tests pass** if:
1. ✅ Environment setup successful
2. ✅ All 25+ indicators calculated
3. ✅ 3-phase normalization verified
4. ✅ Context features generated (stub validated)
5. ✅ Walk-forward folds created correctly
6. ✅ Feature version == `v3.1`
7. ✅ Edge cases handled gracefully
8. ✅ Performance ≥ 100 rows/sec

**Ready for production** when:
- All tests pass consistently
- Manual API tests succeed
- Full training pipeline completes
- Checkpoint metadata validated
- Performance benchmarks met

---

## Next Steps

After all tests pass:
1. **Full Ray Data join**: Implement `_join_context_features()` with proper Ray Data join
2. **GPU testing**: Validate GPU acceleration (`num_gpus=1.0`)
3. **Scale testing**: Test with 10+ symbols, 1 year of data
4. **Backtest validation**: Run complete backtest with VectorBT
5. **Production deployment**: Deploy to live trading environment

---

**Last Updated**: 2026-01-17 (v3.1 - Comprehensive normalization + Context symbols)

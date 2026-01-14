# Validation Tests - Complete Reference

## Overview

Validation tests have been successfully created to validate critical PostgreSQL migration behaviors. These tests ensure that fingerprint-based deduplication and connection pooling work correctly under all conditions.

## Test Files Created

### 1. Fingerprint Deduplication Tests
**File**: [tests/validation/test_fingerprint_deduplication.py](validation/test_fingerprint_deduplication.py)  
**Test Count**: 16 tests  
**Purpose**: Validate fingerprint-based model deduplication system

**Test Classes**:

#### TestFingerprintGeneration (6 tests)
- `test_identical_configs_same_fingerprint` - Identical configs produce same hash
- `test_different_features_different_fingerprint` - Different features produce different hash
- `test_different_hyperparameters_different_fingerprint` - Different hyperparams produce different hash
- `test_different_symbols_different_fingerprint` - Different symbols produce different hash
- `test_order_independent_fingerprint` - Feature list order doesn't affect hash
- `test_fingerprint_format` - Fingerprint is 64-char hex string (SHA256)

#### TestFingerprintDeduplication (5 tests)
- `test_lookup_by_fingerprint_found` - Can find model by fingerprint
- `test_lookup_by_fingerprint_not_found` - Returns None for unknown fingerprint
- `test_duplicate_prevention` - Detects when fingerprint already exists
- `test_different_configs_no_collision` - Different configs don't collide
- `test_multiple_symbols_same_config` - Same config on different symbols creates different fingerprints

#### TestFingerprintEdgeCases (3 tests)
- `test_null_optional_fields` - Null values in optional fields handled consistently
- `test_empty_vs_none_consistency` - Empty lists vs None normalized
- `test_json_serialization_consistency` - JSON serialization doesn't break fingerprint

#### TestFingerprintIntegration (2 tests)
- `test_retrain_creates_duplicate_fingerprint` - Retraining creates same fingerprint
- `test_comprehensive_fingerprint` - All config fields contribute to fingerprint

### 2. Connection Pool Limit Tests
**File**: [tests/validation/test_connection_pool_limits.py](validation/test_connection_pool_limits.py)  
**Test Count**: 15 tests  
**Purpose**: Validate PostgreSQL connection pool behavior and limits

**Test Classes**:

#### TestConnectionPoolLimits (5 tests)
- `test_pool_respects_max_size` - Pool doesn't exceed max_size configuration
- `test_pool_concurrent_acquisitions` - Handles 5 concurrent connections correctly
- `test_pool_reuses_connections` - Connections are reused after release
- `test_pool_under_load` - Handles 20 workers with limited pool (⚠️ SLOW)
- `test_pool_min_size` - Maintains minimum connection count

#### TestConnectionPoolIsolation (2 tests)
- `test_separate_pools_for_services` - Training and Simulation have separate pools
- `test_independent_connections` - Services don't share connections

#### TestConnectionPoolCleanup (2 tests)
- `test_pool_closes_cleanly` - Pool closes without errors
- `test_pool_handles_in_use_connections` - Handles connections still in use during close

#### TestConnectionPoolErrors (2 tests)
- `test_pool_survives_query_errors` - Pool remains functional after query errors
- `test_pool_timeout_configuration` - Timeout configuration is respected

#### TestConnectionPoolPerformance (2 tests)
- `test_pool_acquisition_speed` - Acquisition/release < 10ms (⚠️ SLOW)
- `test_pool_concurrent_throughput` - 50 workers complete < 2s (⚠️ SLOW)

#### TestProcessPoolConnections (1 test)
- `test_each_process_creates_pool` - Each worker process creates own pool

#### TestConnectionPoolRecovery (1 test)
- `test_pool_recreates_failed_connections` - Recovers from connection failures

## Running Validation Tests

### Run All Validation Tests
```bash
# Using test runner
./run_unit_tests.sh validation

# Using pytest directly
pytest tests/validation/ -v

# With coverage
pytest tests/validation/ --cov=training_service --cov=simulation_service --cov-report=term
```

### Run Specific Test Files
```bash
# Fingerprint tests only
pytest tests/validation/test_fingerprint_deduplication.py -v

# Connection pool tests only
pytest tests/validation/test_connection_pool_limits.py -v
```

### Run Specific Test Classes
```bash
# Fingerprint generation tests
pytest tests/validation/test_fingerprint_deduplication.py::TestFingerprintGeneration -v

# Pool limit tests
pytest tests/validation/test_connection_pool_limits.py::TestConnectionPoolLimits -v

# Pool performance tests (slow)
pytest tests/validation/test_connection_pool_limits.py::TestConnectionPoolPerformance -v
```

### Run Individual Tests
```bash
# Test identical configs produce same fingerprint
pytest tests/validation/test_fingerprint_deduplication.py::TestFingerprintGeneration::test_identical_configs_same_fingerprint -v

# Test pool respects max size
pytest tests/validation/test_connection_pool_limits.py::TestConnectionPoolLimits::test_pool_respects_max_size -v
```

### Skip Slow Tests
```bash
# Fast tests only (skip performance benchmarks)
pytest tests/validation/ -m "not slow" -v

# This skips:
# - test_pool_under_load
# - test_pool_acquisition_speed
# - test_pool_concurrent_throughput
```

## Test Coverage

### What Validation Tests Verify

#### Fingerprint System ✅
- ✅ Hash generation is consistent and deterministic
- ✅ Different configurations produce different fingerprints
- ✅ Order-independent hashing (sorted feature lists)
- ✅ Symbol is included in fingerprint calculation
- ✅ Database can look up models by fingerprint
- ✅ Duplicate detection prevents redundant training
- ✅ Edge cases handled (null fields, empty lists, JSON)
- ✅ Integration with training workflow

#### Connection Pools ✅
- ✅ Pool respects max_size limit (doesn't exceed)
- ✅ Concurrent access handled correctly
- ✅ Connections reused efficiently after release
- ✅ Performance under load (20+ concurrent workers)
- ✅ Minimum connection count maintained
- ✅ Service isolation (training vs simulation pools)
- ✅ Clean shutdown and cleanup
- ✅ Error handling (query errors, timeouts)
- ✅ Fast acquisition speed (< 10ms)
- ✅ High throughput (50 workers < 2s)
- ✅ Multi-process support (per-process pools)
- ✅ Connection recovery after failures

## Expected Test Results

### Passing Tests (31/31)
When PostgreSQL is running and configured correctly:
- ✅ All 16 fingerprint tests should pass
- ✅ All 15 connection pool tests should pass
- ⏱️ 3 tests marked slow (5-10s each)
- ⚡ Other tests complete in < 1s each
- 📊 Total runtime: 5-15 seconds (fast), 20-30 seconds (with slow tests)

### Test Output Example
```
tests/validation/test_fingerprint_deduplication.py::TestFingerprintGeneration::test_identical_configs_same_fingerprint PASSED
tests/validation/test_fingerprint_deduplication.py::TestFingerprintGeneration::test_different_features_different_fingerprint PASSED
...
tests/validation/test_connection_pool_limits.py::TestConnectionPoolLimits::test_pool_respects_max_size PASSED
tests/validation/test_connection_pool_limits.py::TestConnectionPoolLimits::test_pool_concurrent_acquisitions PASSED
...

========================= 31 passed in 12.45s =========================
```

## Troubleshooting

### Issue: Tests fail with "function `compute_fingerprint` not found"
**Solution**: Ensure compute_fingerprint is implemented in training_service/pg_db.py:
```python
def compute_fingerprint(config: dict) -> str:
    """Compute SHA256 fingerprint from configuration."""
    import hashlib
    import json
    
    # Normalize config (sort features, handle nulls)
    normalized = {
        "features": sorted(config.get("features", [])),
        "hyperparameters": config.get("hyperparameters", {}),
        "symbol": config.get("symbol", ""),
        "target_col": config.get("target_col", ""),
        # ... other fields
    }
    
    # Compute hash
    config_str = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()
```

### Issue: Connection pool tests fail with "too many connections"
**Solution**: 
1. Check PostgreSQL max connections: `SHOW max_connections;`
2. Close pools between tests (db_tables fixture handles this)
3. Reduce concurrent test workers: `pytest tests/validation/ -n1`

### Issue: Slow tests timeout
**Solution**:
1. Skip slow tests: `pytest tests/validation/ -m "not slow"`
2. Increase timeout: `pytest tests/validation/ --timeout=300`
3. Run slow tests separately: `pytest tests/validation/ -m "slow" -v`

### Issue: Fingerprint collisions in tests
**Solution**: Use unique model IDs and symbols per test:
```python
import uuid
model_data["id"] = f"test-model-{uuid.uuid4()}"
model_data["symbol"] = f"TEST{random.randint(1000, 9999)}"
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Validation Tests

on: [push, pull_request]

jobs:
  validation:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: orchestrator
          POSTGRES_PASSWORD: orchestrator
          POSTGRES_DB: strategy_factory_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements.txt
      
      - name: Run validation tests (fast)
        run: |
          pytest tests/validation/ -v -m "not slow"
      
      - name: Run slow validation tests
        if: github.ref == 'refs/heads/main'
        run: |
          pytest tests/validation/ -v -m "slow"
```

## Documentation

### Test Documentation Files
- **VALIDATION_TESTS.md** - This file (comprehensive guide)
- **TEST_SUITE_SUMMARY.md** - Overall test suite overview
- **count_validation_tests.py** - Test statistics script
- **list_tests.py** - Interactive test catalog

### Quick Reference
```bash
# List all validation tests
python tests/list_tests.py | grep -A50 "VALIDATION TESTS"

# Count validation tests
python tests/count_validation_tests.py

# View test details
pytest tests/validation/ --collect-only
```

## Test Metrics

### Coverage Goals
- **Fingerprint System**: 95%+ coverage of compute_fingerprint and lookup logic
- **Connection Pools**: 90%+ coverage of pool creation, acquisition, release, cleanup
- **Edge Cases**: 100% coverage of null handling, serialization, error recovery

### Performance Benchmarks
- **Fingerprint Generation**: < 1ms per fingerprint
- **Database Lookup**: < 5ms per lookup
- **Pool Acquisition**: < 10ms per acquire/release
- **Concurrent Throughput**: 50 workers < 2s
- **Pool Under Load**: 20 workers complete successfully

## Next Steps

After validation tests pass:
1. ✅ Unit tests (53 tests) - Complete
2. ✅ API tests (39 tests) - Complete
3. ✅ Integration tests (22 tests) - Complete
4. ✅ **Validation tests (31 tests) - Complete** ← YOU ARE HERE
5. ⏳ Load tests (parallel training, concurrent DB access)
6. ⏳ Performance tests (CPU utilization, memory usage)
7. ⏳ Error handling tests (DB failures, process pool errors)
8. ⏳ CI/CD setup (automated test runs)

---

**Status**: ✅ Validation tests complete (31 tests created)  
**Next**: Load tests and performance tests  
**Total Test Count**: 145 tests (53 unit + 39 API + 22 integration + 31 validation)

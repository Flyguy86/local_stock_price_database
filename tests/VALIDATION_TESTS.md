# Validation Tests Quick Start Guide

## Overview

Validation tests verify critical system behaviors and constraints. Unlike unit tests (test individual components) or integration tests (test workflows), validation tests ensure that important business rules and technical constraints are properly enforced.

## Test Coverage

### Fingerprint Deduplication (`test_fingerprint_deduplication.py`)
**Purpose**: Validate that the fingerprinting system correctly identifies duplicate models and prevents redundant training.

**Test Categories**:
- ✅ **Fingerprint Generation** (6 tests)
  - Identical configs produce identical fingerprints
  - Different features/hyperparameters produce different fingerprints
  - Different symbols produce different fingerprints
  - Feature order doesn't affect fingerprint (normalized)
  - JSON serialization consistency
  
- ✅ **Deduplication Logic** (5 tests)
  - Lookup by fingerprint (found/not found)
  - Duplicate prevention detection
  - Different configs don't collide
  - Multiple symbols with same config
  
- ✅ **Edge Cases** (3 tests)
  - Null optional fields handled correctly
  - Empty lists vs None consistency
  - JSON serialization consistency
  
- ✅ **Integration** (2 tests)
  - Retrain creates duplicate fingerprint
  - Comprehensive fingerprint with all fields

### Connection Pool Limits (`test_connection_pool_limits.py`)
**Purpose**: Validate that PostgreSQL connection pools behave correctly under various conditions and respect configured limits.

**Test Categories**:
- ✅ **Pool Limits** (5 tests)
  - Respects max_size configuration
  - Handles concurrent acquisitions
  - Connection reuse after release
  - Performance under load (slow)
  - Maintains minimum connection count
  
- ✅ **Pool Isolation** (2 tests)
  - Separate pools for training/simulation services
  - Independent connections per service
  
- ✅ **Cleanup** (2 tests)
  - Pool closes cleanly
  - Handles connections in use during close
  
- ✅ **Error Handling** (2 tests)
  - Survives query errors
  - Timeout configuration
  
- ✅ **Performance** (2 tests)
  - Fast acquisition speed (slow)
  - High concurrent throughput (slow)
  
- ✅ **Multi-Process** (1 test)
  - Each process creates own pool
  
- ✅ **Recovery** (1 test)
  - Recreates failed connections

## Running Validation Tests

### Prerequisites

1. **PostgreSQL must be running**:
   ```bash
   docker compose up postgres -d
   ```

2. **Install test dependencies**:
   ```bash
   pip install -r tests/requirements.txt
   ```

### Run All Validation Tests

```bash
# Run all validation tests
pytest tests/validation/ -v

# Run fingerprint tests only
pytest tests/validation/test_fingerprint_deduplication.py -v

# Run connection pool tests only
pytest tests/validation/test_connection_pool_limits.py -v

# Skip slow tests
pytest tests/validation/ -m "not slow" -v
```

### Run Specific Test Classes

```bash
# Test fingerprint generation
pytest tests/validation/test_fingerprint_deduplication.py::TestFingerprintGeneration -v

# Test pool limits
pytest tests/validation/test_connection_pool_limits.py::TestConnectionPoolLimits -v

# Test deduplication logic
pytest tests/validation/test_fingerprint_deduplication.py::TestFingerprintDeduplication -v
```

### Run Individual Tests

```bash
# Test identical configs produce same fingerprint
pytest tests/validation/test_fingerprint_deduplication.py::TestFingerprintGeneration::test_identical_configs_same_fingerprint -v

# Test pool respects max size
pytest tests/validation/test_connection_pool_limits.py::TestConnectionPoolLimits::test_pool_respects_max_size -v
```

## Test Structure

Validation tests follow assertion-heavy patterns:

```python
class TestValidationScenario:
    """Validate specific behavior."""
    
    @pytest.mark.asyncio
    async def test_constraint_enforced(self, db_tables):
        """
        Validation: System enforces important constraint.
        
        Detailed explanation of what's being validated and why.
        """
        # Setup: Create test data
        # ...
        
        # Action: Perform operation that should trigger validation
        # ...
        
        # Assert: Verify constraint is enforced
        assert expected_behavior
        assert not forbidden_behavior
```

## Expected Test Results

### Fingerprint Deduplication (16 tests)
- ✅ Fingerprint generation (6 tests)
- ✅ Deduplication logic (5 tests)
- ✅ Edge cases (3 tests)
- ✅ Integration (2 tests)

### Connection Pool Limits (15 tests)
- ✅ Pool limits (5 tests)
- ✅ Pool isolation (2 tests)
- ✅ Cleanup (2 tests)
- ✅ Error handling (2 tests)
- ✅ Performance (2 tests, marked slow)
- ✅ Multi-process (1 test)
- ✅ Recovery (1 test)

**Total Validation Tests**: ~31 tests  
**Expected Runtime**: 5-15 seconds (20-30 seconds with slow tests)

## What Validation Tests Check

### Business Logic Validation
- ✅ **Fingerprint uniqueness**: Same configuration always produces same hash
- ✅ **Duplicate detection**: System can identify when a model already exists
- ✅ **Configuration sensitivity**: Changes to important fields change fingerprint
- ✅ **Normalization**: Order-independent hashing (sorted features)

### Technical Constraint Validation
- ✅ **Pool size limits**: Connection pools don't exceed max_size
- ✅ **Resource isolation**: Each service has independent pool
- ✅ **Concurrent safety**: Pool handles concurrent requests correctly
- ✅ **Cleanup**: Connections released properly
- ✅ **Error recovery**: Pool survives errors and recovers

### Data Integrity Validation
- ✅ **JSON consistency**: Serialization/deserialization preserves fingerprints
- ✅ **Null handling**: Optional fields handled consistently
- ✅ **Type consistency**: Empty lists vs None normalized

## Common Issues & Solutions

### Issue: "fingerprint not found" in tests
**Solution**: Ensure fingerprint is computed and set before creating model:
```python
fingerprint = compute_fingerprint(config)
model_data["fingerprint"] = fingerprint
await db.create_model_record(model_data)
```

### Issue: Pool tests fail with "too many connections"
**Solution**: Close pools between tests or use db_tables fixture:
```python
@pytest.mark.asyncio
async def test_something(self, db_tables):  # ← Ensures cleanup
    ...
```

### Issue: Slow validation tests timeout
**Solution**: Skip slow tests or increase timeout:
```bash
pytest tests/validation/ -m "not slow"  # Skip slow tests
pytest tests/validation/ --timeout=300  # 5 minute timeout
```

### Issue: Fingerprint collision between tests
**Solution**: Use unique model IDs and symbols per test:
```python
model_data["id"] = f"test-model-{uuid.uuid4()}"
model_data["symbol"] = f"TEST{random.randint(1000, 9999)}"
```

## Validation vs Other Test Types

| Aspect | Unit Tests | API Tests | Integration Tests | Validation Tests |
|--------|------------|-----------|-------------------|------------------|
| **Purpose** | Component logic | HTTP interface | End-to-end flow | Constraints/rules |
| **Scope** | Single function | Single endpoint | Complete workflow | Specific behavior |
| **Example** | Test DB insert | Test /models GET | Test train→save→query | Test fingerprint dedup |
| **Focus** | Code correctness | API contract | Data flow | Business rules |

## Performance Considerations

Validation tests vary in speed:
- **Fast** (< 1s): Fingerprint generation, basic pool operations
- **Standard** (1-5s): Deduplication checks, pool concurrency
- **Slow** (5-30s): Performance benchmarks, load testing (marked with `@pytest.mark.slow`)

To optimize test runs:
```bash
# Fast tests only (for rapid iteration)
pytest tests/validation/ -m "not slow" -v

# All tests (for comprehensive validation)
pytest tests/validation/ -v
```

## Integration with CI/CD

Validation tests should run in CI/CD after unit/API tests:

```yaml
# .github/workflows/test.yml
- name: Run Validation Tests
  run: |
    docker compose up postgres -d
    sleep 5
    pytest tests/validation/ -v -m "not slow"
  
- name: Run Slow Validation Tests
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  run: |
    pytest tests/validation/ -v -m "slow"
```

## Debugging Validation Tests

### Check fingerprint computation
```python
from training_service.pg_db import compute_fingerprint

config = {
    "features": ["rsi_14", "sma_20"],
    "hyperparameters": {"alpha": 1.0},
    "symbol": "AAPL",
    "target_col": "close"
}

print(compute_fingerprint(config))
# Expected: 64-character hex string
```

### Check pool status
```python
from training_service.pg_db import get_pool

pool = await get_pool()
print(f"Pool size: {pool.get_size()}")
print(f"Min size: {pool.get_min_size()}")
print(f"Max size: {pool.get_max_size()}")
```

### View database state
```bash
# Connect to database
psql -U orchestrator -d strategy_factory_test -h localhost

# Check for duplicate fingerprints
SELECT fingerprint, COUNT(*) 
FROM models 
GROUP BY fingerprint 
HAVING COUNT(*) > 1;

# Check connection count
SELECT COUNT(*) FROM pg_stat_activity 
WHERE datname = 'strategy_factory_test';
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pytest tests/validation/ -v` | Run all validation tests |
| `pytest tests/validation/ -m "not slow"` | Skip slow tests |
| `pytest tests/validation/ -k fingerprint` | Run fingerprint tests |
| `pytest tests/validation/ -k pool` | Run connection pool tests |
| `pytest tests/validation/ --collect-only` | List all validation tests |
| `pytest tests/validation/ -x` | Stop on first failure |

---

**Total Validation Test Count**: ~31 tests  
**Expected Pass Rate**: 100% (with PostgreSQL running)  
**Average Runtime**: 5-15 seconds (fast), 20-30 seconds (with slow tests)

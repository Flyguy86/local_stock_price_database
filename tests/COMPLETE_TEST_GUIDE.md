# Complete Test Suite Documentation

## Overview

The PostgreSQL migration includes a comprehensive test suite with **200+ tests** covering all aspects of the system. Tests are organized into categories for efficient execution and clear reporting.

## Test Categories

### 1. Unit Tests (53 tests)
**Location**: `tests/unit/`  
**Purpose**: Test individual components in isolation  
**Runtime**: ~5-10 seconds

- **PostgreSQL Database Layer** (30 tests) - `test_pg_db.py`
  - Table creation and schema validation
  - CRUD operations (Create, Read, Update, Delete)
  - Fingerprint-based deduplication
  - Status transitions and validation
  - Feature importance storage
  - Cascade delete operations
  
- **Sync DB Wrapper** (8 tests) - `test_sync_wrapper.py`
  - Multi-process model creation
  - Concurrent updates from different processes
  - Process isolation with separate connection pools
  - Performance under load
  
- **Process Pool Executor** (15 tests) - `test_process_pool.py`
  - Parallel task execution
  - Worker PID verification
  - Max workers limit enforcement
  - Worker recycling (max_tasks_per_child)
  - Database access from workers
  - Speedup benchmarks

### 2. API Tests (39 tests)
**Location**: `tests/api/`  
**Purpose**: Test HTTP API endpoints  
**Runtime**: ~10-15 seconds

- **Training Service API** (21 tests) - `test_training_api.py`
  - Health and logging endpoints
  - Data availability endpoints
  - Algorithm discovery
  - Model management (list, get, delete)
  - Training operations (train, batch, retrain)
  - Feature importance retrieval
  - Transfer learning
  
- **Simulation Service API** (18 tests) - `test_simulation_api.py`
  - Health and logging endpoints
  - Configuration endpoints
  - Simulation history management
  - Simulation execution (single, batch)
  - Bot training
  - Performance metrics

### 3. Integration Tests (22 tests)
**Location**: `tests/integration/`  
**Purpose**: Test complete end-to-end workflows  
**Runtime**: ~15-25 seconds

- **Training Workflows** (11 tests) - `test_training_workflow.py`
  - Complete training pipeline (data → train → save → query)
  - Batch training (multiple models)
  - Parent-child training (transfer learning)
  - Model retraining workflow
  - Feature importance storage/retrieval
  - Model deletion with cascade
  - Concurrent training jobs
  
- **Simulation Workflows** (11 tests) - `test_simulation_workflow.py`
  - Complete simulation pipeline (model → simulate → analyze)
  - Batch simulations with ranking
  - Pagination for large result sets
  - History deletion workflow
  - Performance metrics validation
  - Multiple strategy comparison
  - Cross-symbol testing

### 4. Validation Tests (31 tests)
**Location**: `tests/validation/`  
**Purpose**: Validate critical system behaviors and constraints  
**Runtime**: ~10-20 seconds (30s with slow tests)

- **Fingerprint Deduplication** (16 tests) - `test_fingerprint_deduplication.py`
  - Fingerprint generation consistency
  - Identical configs → same fingerprint
  - Different configs → different fingerprints
  - Order-independent hashing
  - Database lookup by fingerprint
  - Duplicate prevention
  - Edge cases (null fields, JSON)
  
- **Connection Pool Limits** (15 tests) - `test_connection_pool_limits.py`
  - Pool respects max_size configuration
  - Concurrent connection handling
  - Connection reuse after release
  - Service isolation (training/simulation)
  - Clean shutdown and cleanup
  - Error handling and recovery
  - Performance benchmarks

### 5. Load Tests (25+ tests)
**Location**: `tests/load/`  
**Purpose**: Test system under high concurrent load  
**Runtime**: ~30-60 seconds

- **Parallel Training** (6 tests) - `test_parallel_training.py`
  - 8 parallel training jobs
  - 16 parallel training jobs
  - Parallel status updates
  - Parallel feature importance storage
  - Parallel model deletion
  
- **Process Pool Load** (3 tests) - `test_parallel_training.py`
  - 8 concurrent workers
  - Process pool throughput
  - Error handling under load
  
- **Database Contention** (3 tests) - `test_parallel_training.py`
  - Concurrent reads (no blocking)
  - Mixed read/write operations
  - Bulk insert performance
  
- **Concurrent Database Access** (6 tests) - `test_concurrent_database.py`
  - 100 concurrent queries
  - Concurrent writes to different tables
  - Concurrent updates to same model
  - Simulation concurrent history writes
  - Paginated queries with concurrent inserts
  
- **Connection Pool Stress** (4 tests) - `test_concurrent_database.py`
  - Connection pool saturation
  - Rapid acquire/release cycles
  - Mixed query complexity
  - Connection timeout handling

### 6. Performance Tests (13+ tests)
**Location**: `tests/performance/`  
**Purpose**: Monitor resource usage and performance  
**Runtime**: ~40-80 seconds

- **CPU Utilization** (3 tests) - `test_cpu_memory.py`
  - Parallel training CPU usage
  - Database query CPU efficiency
  - Process pool CPU distribution
  
- **Memory Usage** (4 tests) - `test_cpu_memory.py`
  - Memory during bulk insert
  - Memory leak detection
  - Connection pool memory usage
  - Large result set memory
  
- **Query Performance** (2 tests) - `test_cpu_memory.py`
  - Query response times (avg, p95)
  - Bulk query throughput (QPS)

### 7. Error Handling Tests (25+ tests)
**Location**: `tests/error_handling/`  
**Purpose**: Test error scenarios and recovery  
**Runtime**: ~20-40 seconds

- **Database Connection Failures** (9 tests) - `test_database_failures.py`
  - Connection failure during query
  - Pool exhaustion recovery
  - Query timeout handling
  - Invalid SQL error handling
  - Foreign key violations
  - Duplicate key errors
  - Transaction rollback
  
- **Database Recovery** (3 tests) - `test_database_failures.py`
  - Reconnect after connection lost
  - Pool graceful degradation
  - Retry on transient failure
  
- **Data Validation** (3 tests) - `test_database_failures.py`
  - Invalid status transitions
  - NULL value validation
  - Invalid JSON data
  
- **Process Pool Failures** (5 tests) - `test_process_pool_failures.py`
  - Worker exception handling
  - Worker timeout handling
  - Worker crash recovery
  - Max workers enforcement
  - Worker memory limits
  
- **Resource Management** (3 tests) - `test_process_pool_failures.py`
  - Process pool cleanup on exit
  - Database connection cleanup per worker
  - SIGTERM handling
  
- **Error Propagation** (2 tests) - `test_process_pool_failures.py`
  - Exception traceback preservation
  - Multiple simultaneous failures

## Quick Start

### Run All Tests
```bash
# Using test runner script
./run_unit_tests.sh

# Using pytest directly
pytest tests/ -v
```

### Run by Category
```bash
./run_unit_tests.sh unit           # Unit tests only
./run_unit_tests.sh api            # API tests only
./run_unit_tests.sh integration    # Integration tests only
./run_unit_tests.sh validation     # Validation tests only
./run_unit_tests.sh fast           # Fast tests (skip slow)

# Load, performance, and error handling tests
pytest tests/load/ -v
pytest tests/performance/ -v
pytest tests/error_handling/ -v
```

### Run with Coverage
```bash
./run_unit_tests.sh --coverage

# Or with pytest
pytest tests/ --cov=training_service --cov=simulation_service --cov-report=html
```

## Test Statistics

| Category | Test Count | Runtime | Files |
|----------|------------|---------|-------|
| Unit Tests | 53 | 5-10s | 3 |
| API Tests | 39 | 10-15s | 2 |
| Integration Tests | 22 | 15-25s | 2 |
| Validation Tests | 31 | 10-30s | 2 |
| Load Tests | 25+ | 30-60s | 2 |
| Performance Tests | 13+ | 40-80s | 1 |
| Error Handling Tests | 25+ | 20-40s | 2 |
| **Total** | **208+** | **130-260s** | **14** |

## Test Markers

Tests use pytest markers for selective execution:

```python
@pytest.mark.unit           # Unit test
@pytest.mark.api            # API test
@pytest.mark.integration    # Integration test
@pytest.mark.load           # Load test
@pytest.mark.performance    # Performance test
@pytest.mark.error_handling # Error handling test
@pytest.mark.slow           # Slow running test
@pytest.mark.asyncio        # Async test
```

### Run by Marker
```bash
pytest tests/ -m unit                    # Unit tests only
pytest tests/ -m "not slow"              # Skip slow tests
pytest tests/ -m "load or performance"   # Load + performance
pytest tests/ -m "integration and not slow"  # Fast integration tests
```

## CI/CD Integration

### GitHub Actions Workflow
**File**: `.github/workflows/test.yml`

The test suite runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs**:
1. **Unit Tests** - Always runs first
2. **API Tests** - Runs after unit tests
3. **Integration Tests** - Runs after API tests (skips slow)
4. **Validation Tests** - Runs after integration (skips slow)
5. **Load & Performance Tests** - Only on main branch pushes
6. **Error Handling Tests** - Runs with validation
7. **Coverage Report** - Generates combined coverage
8. **Test Summary** - Aggregates all results

### Local CI Simulation
```bash
# Simulate CI pipeline locally
pytest tests/unit/ -v --tb=short
pytest tests/api/ -v --tb=short
pytest tests/integration/ -v -m "not slow"
pytest tests/validation/ -v -m "not slow"
pytest tests/error_handling/ -v
```

## Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| Overall | 85% | TBD |
| Core Modules | 90% | TBD |
| training_service.pg_db | 95% | TBD |
| training_service.db | 90% | TBD |
| simulation_service.database | 90% | TBD |

## Test Infrastructure

### Fixtures (conftest.py)
- `test_db_pool` - PostgreSQL connection pool (session scope)
- `db_tables` - Fresh database tables per test (auto-cleanup)
- `sample_model_data` - Generate mock model records
- `training_db` - TrainingDB instance
- `simulation_db` - SimulationDB instance
- `api_client` - AsyncClient for HTTP testing
- `mock_data_available` - Mock data availability

### Configuration (pytest.ini)
- Test discovery patterns
- Marker definitions
- Coverage settings
- Output formatting

### Utilities
- `list_tests.py` - Interactive test catalog
- `count_tests.py` - Test statistics
- `count_validation_tests.py` - Validation test counts
- `count_integration_tests.py` - Integration test counts

## Documentation Files

- **TEST_SUITE_SUMMARY.md** - This file (complete overview)
- **UNIT_TESTS.md** - Unit test quick start guide
- **API_TESTS.md** - API test quick start guide  
- **INTEGRATION_TESTS.md** - Integration test quick start guide
- **VALIDATION_TESTS.md** - Validation test quick start guide
- **VALIDATION_TESTS_REFERENCE.md** - Detailed validation reference

## Common Commands

```bash
# List all tests
pytest tests/ --collect-only

# Run specific file
pytest tests/unit/test_pg_db.py -v

# Run specific class
pytest tests/api/test_training_api.py::TestModelManagement -v

# Run specific test
pytest tests/unit/test_pg_db.py::TestModelCRUD::test_create_model -v

# Run tests matching pattern
pytest tests/ -k "fingerprint" -v

# Stop on first failure
pytest tests/ -x

# Rerun last failures
pytest tests/ --lf

# Show test durations
pytest tests/ --durations=10

# Parallel execution (if pytest-xdist installed)
pytest tests/ -n auto
```

## Troubleshooting

### PostgreSQL Connection Issues
```bash
# Verify PostgreSQL is running
docker compose ps postgres

# Check connection
psql -U orchestrator -d strategy_factory_test -h localhost

# Reset database
docker compose down postgres
docker compose up postgres -d
sleep 5
pytest tests/ -v
```

### Test Failures
```bash
# Verbose output with full traceback
pytest tests/ -vv --tb=long

# Show print statements
pytest tests/ -s

# Run specific failing test
pytest tests/path/to/test.py::TestClass::test_method -vv
```

### Performance Issues
```bash
# Profile slow tests
pytest tests/ --durations=20

# Skip slow tests
pytest tests/ -m "not slow"

# Run subset
pytest tests/unit/ tests/api/  # Skip integration/load/performance
```

## Best Practices

1. **Run fast tests frequently** - `./run_unit_tests.sh fast`
2. **Run full suite before PR** - `./run_unit_tests.sh --coverage`
3. **Check coverage** - Aim for 85%+ overall
4. **Write descriptive test names** - Explain what's being tested
5. **Use fixtures** - Avoid test data duplication
6. **Mark slow tests** - Use `@pytest.mark.slow`
7. **Clean up resources** - Use fixtures for setup/teardown
8. **Isolate tests** - Each test should be independent
9. **Test edge cases** - Don't just test happy path
10. **Document complex tests** - Add docstrings explaining why

## Next Steps

1. Run initial test suite to establish baseline coverage
2. Add missing tests based on coverage report
3. Integrate with CI/CD (already set up)
4. Monitor test performance over time
5. Add regression tests for bug fixes
6. Expand load tests for production scenarios
7. Add chaos engineering tests (network failures, etc.)

---

**Total Test Coverage**: 208+ tests across 7 categories  
**Estimated Full Runtime**: 130-260 seconds (2-4 minutes)  
**CI/CD Runtime**: 60-90 seconds (fast tests only)

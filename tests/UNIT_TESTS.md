# Unit Tests - Quick Start Guide

## Installation

```bash
# Install test dependencies
pip install -r tests/requirements.txt
```

## Running Tests

### All Unit Tests
```bash
# Run all unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ -v --cov=training_service --cov=simulation_service --cov-report=html
```

### Individual Test Files
```bash
# PostgreSQL database tests
pytest tests/unit/test_pg_db.py -v

# Sync wrapper tests
pytest tests/unit/test_sync_wrapper.py -v

# Process pool tests
pytest tests/unit/test_process_pool.py -v
```

### Specific Test Classes or Functions
```bash
# Run specific test class
pytest tests/unit/test_pg_db.py::TestPostgreSQLDatabaseLayer -v

# Run specific test function
pytest tests/unit/test_pg_db.py::TestPostgreSQLDatabaseLayer::test_ensure_tables -v
```

### With Options
```bash
# Run slow tests (performance benchmarks)
pytest tests/unit/ -v --run-slow

# Stop on first failure
pytest tests/unit/ -v -x

# Show print statements
pytest tests/unit/ -v -s

# Run in parallel (requires pytest-xdist)
pytest tests/unit/ -v -n auto
```

## Prerequisites

1. **PostgreSQL Running**: Tests require PostgreSQL to be running
   ```bash
   docker compose up postgres -d
   ```

2. **Test Database**: Tests will create/use `strategy_factory_test` database
   - Automatically created on first run
   - Cleaned between tests

3. **Environment Variables** (optional):
   ```bash
   export TEST_POSTGRES_URL="postgresql://orchestrator:orchestrator_secret@localhost:5432/strategy_factory_test"
   ```

## Test Organization

```
tests/unit/
├── test_pg_db.py           # PostgreSQL database layer tests (30 tests)
├── test_sync_wrapper.py    # Sync wrapper tests (8 tests)
└── test_process_pool.py    # Process pool executor tests (15 tests)
```

## What's Tested

### test_pg_db.py (30 tests)
- ✅ Table creation and schema validation
- ✅ Model CRUD operations (create, read, update, delete)
- ✅ Fingerprint-based lookups
- ✅ Status transitions
- ✅ Feature importance storage
- ✅ Cascade deletes
- ✅ Simulation history operations
- ✅ Top strategies with pagination

### test_sync_wrapper.py (8 tests)
- ✅ Wrapper initialization
- ✅ Basic CRUD through sync wrapper
- ✅ Multi-process model creation
- ✅ Concurrent updates from different processes
- ✅ Process isolation (separate connection pools)
- ✅ Performance under load

### test_process_pool.py (15 tests)
- ✅ Pool creation and configuration
- ✅ Parallel task execution
- ✅ Different PIDs verification
- ✅ Max workers limit enforcement
- ✅ Error handling
- ✅ Graceful shutdown
- ✅ Worker recycling (max_tasks_per_child)
- ✅ Database access from workers
- ✅ Speedup measurement (slow test)

## Expected Results

### All Tests
```
tests/unit/test_pg_db.py ............................ [ 56%]
tests/unit/test_sync_wrapper.py ......                [ 72%]
tests/unit/test_process_pool.py ...........          [100%]

================= 53 passed in 12.34s =================
```

### With Coverage
```
---------- coverage: platform linux, python 3.11 -----------
Name                               Stmts   Miss  Cover
------------------------------------------------------
training_service/pg_db.py            245     12    95%
training_service/sync_db_wrapper.py   85      4    95%
simulation_service/pg_db.py          178      8    95%
------------------------------------------------------
TOTAL                                508     24    95%
```

## Troubleshooting

### PostgreSQL Connection Error
```
Error: could not connect to server
```
**Solution**: Start PostgreSQL
```bash
docker compose up postgres -d
docker compose logs postgres | grep "ready to accept connections"
```

### Import Errors
```
ModuleNotFoundError: No module named 'training_service'
```
**Solution**: Ensure running from project root
```bash
cd /workspaces/local_stock_price_database
pytest tests/unit/ -v
```

### Async Test Errors
```
RuntimeError: no running event loop
```
**Solution**: Ensure pytest-asyncio is installed
```bash
pip install pytest-asyncio
```

### Database Permission Errors
```
asyncpg.exceptions.InsufficientPrivilegeError
```
**Solution**: Check database credentials
```bash
# Verify connection
docker compose exec postgres psql -U orchestrator -d strategy_factory_test -c "SELECT 1;"
```

## Next Steps

After unit tests pass:
1. Run API tests: `pytest tests/api/ -v`
2. Run integration tests: `pytest tests/integration/ -v`
3. Run all tests: `pytest tests/ -v --cov`

## CI/CD Integration

```yaml
# .github/workflows/test.yml
- name: Run unit tests
  run: |
    pip install -r tests/requirements.txt
    pytest tests/unit/ -v --cov --cov-report=xml
```

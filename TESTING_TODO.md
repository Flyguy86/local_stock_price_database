# Testing & Validation TODO List

## Overview
Comprehensive testing plan to validate PostgreSQL migration, async architecture, and multi-CPU parallelism.

---

## 🧪 Unit Tests

### 1. ✅ PostgreSQL Database Layer (`tests/test_pg_db.py`)

**Test training_service/pg_db.py:**
- [ ] `test_ensure_tables()` - Verify all tables created with correct schema
- [ ] `test_create_model_record()` - Insert model with all fingerprint fields
- [ ] `test_get_model()` - Retrieve model by ID
- [ ] `test_get_model_by_fingerprint()` - Fingerprint-based lookup
- [ ] `test_update_model_status()` - Status transitions (preprocessing → training → completed)
- [ ] `test_list_models()` - Pagination and filtering
- [ ] `test_delete_model()` - Cascade delete (model + features_log)
- [ ] `test_connection_pool()` - Min/max connections respected

**Test simulation_service/pg_db.py:**
- [ ] `test_save_simulation_history()` - Insert simulation result
- [ ] `test_get_simulation_history()` - Retrieve recent simulations
- [ ] `test_get_top_strategies()` - Pagination and SQN sorting
- [ ] `test_delete_all_history()` - Bulk delete

**Commands:**
```bash
pytest tests/test_pg_db.py -v
pytest tests/test_pg_db.py::test_ensure_tables -v
```

---

### 2. ✅ Sync DB Wrapper (`tests/test_sync_wrapper.py`)

**Test training_service/sync_db_wrapper.py:**
- [ ] `test_wrapper_creates_pool_per_process()` - Each process gets own pool
- [ ] `test_update_model_status_from_sync_context()` - Sync wrapper works
- [ ] `test_multiple_processes_isolated_pools()` - No shared state
- [ ] `test_wrapper_cleanup_on_process_exit()` - Pools close cleanly

**Commands:**
```bash
pytest tests/test_sync_wrapper.py -v
```

---

### 3. ✅ Process Pool Executor (`tests/test_process_pool.py`)

**Test training_service/main.py process pool:**
- [ ] `test_process_pool_creation()` - Pool created with CPU_COUNT workers
- [ ] `test_submit_training_task()` - Tasks submitted to pool
- [ ] `test_parallel_execution()` - Multiple tasks run simultaneously
- [ ] `test_different_pids()` - Tasks run in different processes
- [ ] `test_max_tasks_per_child()` - Workers recycled after N tasks
- [ ] `test_pool_shutdown()` - Graceful shutdown on service stop

**Commands:**
```bash
pytest tests/test_process_pool.py -v
python test_process_pool.py  # Standalone test
```

---

## 🌐 API Tests

### 4. ✅ Training Service Endpoints (`tests/test_training_api.py`)

**Test all FastAPI endpoints:**
- [ ] `GET /health` - Returns healthy status
- [ ] `GET /models` - Returns list of models
- [ ] `GET /models/{id}` - Returns specific model
- [ ] `POST /train` - Creates training job, returns ID
- [ ] `POST /train/batch` - Creates 4 training jobs
- [ ] `POST /retrain/{id}` - Creates retrain job
- [ ] `POST /api/train_with_parent` - Creates child model with feature whitelist
- [ ] `GET /api/model/{id}/importance` - Returns feature importance
- [ ] `GET /api/model/{id}/config` - Returns model config for fingerprint
- [ ] `DELETE /models/{id}` - Deletes model
- [ ] `DELETE /models/all` - Deletes all models

**Validation:**
- [ ] Response schemas match expectations
- [ ] Status codes correct (200, 404, 500)
- [ ] Training jobs transition through statuses
- [ ] Error messages clear and actionable

**Commands:**
```bash
pytest tests/test_training_api.py -v
pytest tests/test_training_api.py::test_train_endpoint -v
```

---

### 5. ✅ Simulation Service Endpoints (`tests/test_simulation_api.py`)

**Test all FastAPI endpoints:**
- [ ] `GET /health` - Returns healthy status with model/ticker counts
- [ ] `GET /api/config` - Returns available models and tickers
- [ ] `GET /api/history?limit=50` - Returns simulation history
- [ ] `GET /history/top?limit=15&offset=0` - Returns paginated top strategies
- [ ] `POST /api/simulate` - Runs simulation, saves to PostgreSQL
- [ ] `POST /api/batch_simulate` - Runs multiple simulations
- [ ] `POST /api/train_bot` - Trains trading bot
- [ ] `DELETE /history/all` - Deletes all simulation history
- [ ] `GET /logs` - Returns recent logs

**Validation:**
- [ ] Simulation results saved to PostgreSQL
- [ ] Top strategies sorted by SQN descending
- [ ] Pagination works correctly

**Commands:**
```bash
pytest tests/test_simulation_api.py -v
```

---

## 🔗 Integration Tests

### 6. ✅ Train → Save → Query Flow (`tests/integration/test_training_flow.py`)

**End-to-end training workflow:**
- [ ] Submit `/train` request → verify job created
- [ ] Poll model status until "completed" or "failed"
- [ ] Verify model record in PostgreSQL with all fields
- [ ] Verify feature importance in `features_log` table
- [ ] Query `/models` endpoint → verify model appears
- [ ] Query `/api/model/{id}/config` → verify fingerprint fields
- [ ] Delete model → verify cascade delete of features_log

**Commands:**
```bash
pytest tests/integration/test_training_flow.py -v
```

---

### 7. ✅ Simulate → Save History Flow (`tests/integration/test_simulation_flow.py`)

**End-to-end simulation workflow:**
- [ ] Get available models from `/api/config`
- [ ] Submit `/api/simulate` request
- [ ] Verify simulation runs and returns stats
- [ ] Query `/api/history` → verify simulation saved
- [ ] Query `/history/top` → verify appears in top strategies
- [ ] Submit multiple simulations for same model
- [ ] Verify pagination works with offset

**Commands:**
```bash
pytest tests/integration/test_simulation_flow.py -v
```

---

## 📈 Load Tests

### 8. ✅ Parallel Training (`tests/load/test_parallel_training.py`)

**Test multi-CPU parallelism:**
- [ ] Submit 8 training jobs simultaneously
- [ ] Monitor process creation (should spawn CPU_COUNT workers)
- [ ] Verify tasks run with different PIDs
- [ ] Measure total execution time (should be ~1/8 of sequential)
- [ ] Verify all jobs complete successfully
- [ ] Check for memory leaks after 100+ jobs

**Metrics to capture:**
- [ ] Number of concurrent processes
- [ ] CPU utilization (should approach 100% × cores)
- [ ] Time to complete N jobs
- [ ] Memory usage per worker

**Commands:**
```bash
pytest tests/load/test_parallel_training.py -v
python tests/load/parallel_training_benchmark.py  # Standalone benchmark
```

---

### 9. ✅ Concurrent Database Access (`tests/load/test_concurrent_db.py`)

**Test PostgreSQL under load:**
- [ ] 10 concurrent writes to `models` table
- [ ] 50 concurrent reads from `models` table
- [ ] 100 concurrent simulation history writes
- [ ] Mixed read/write operations (50 each)
- [ ] Verify no connection pool exhaustion
- [ ] Verify no deadlocks or race conditions

**Metrics:**
- [ ] Connection pool usage (should not exceed max_size)
- [ ] Query latency under load
- [ ] Error rate (should be 0%)

**Commands:**
```bash
pytest tests/load/test_concurrent_db.py -v
```

---

## ✅ Validation Tests

### 10. ✅ Fingerprint Deduplication (`tests/validation/test_fingerprint.py`)

**Test model deduplication:**
- [ ] Create two models with identical configs → same fingerprint
- [ ] Verify fingerprint changes when features change
- [ ] Verify fingerprint changes when hyperparameters change
- [ ] Verify fingerprint changes when timeframe changes
- [ ] Verify fingerprint changes when train_window changes
- [ ] Test all 12+ fingerprint parameters affect hash
- [ ] Query `get_model_by_fingerprint()` → verify finds duplicates

**Commands:**
```bash
pytest tests/validation/test_fingerprint.py -v
python validate_migration.py  # Includes fingerprint tests
```

---

### 11. ✅ Connection Pool Limits (`tests/validation/test_connection_pool.py`)

**Test connection pooling:**
- [ ] Verify pool creates min_size=2 connections on startup
- [ ] Verify pool doesn't exceed max_size=10 connections
- [ ] Verify connections released after queries
- [ ] Test pool behavior when max_size reached
- [ ] Verify pool recovers from connection failures
- [ ] Test command_timeout=60 enforced

**Commands:**
```bash
pytest tests/validation/test_connection_pool.py -v
```

---

## ⚡ Performance Tests

### 12. ✅ CPU Utilization (`tests/performance/test_cpu_usage.py`)

**Measure CPU efficiency:**
- [ ] Baseline: Single training job CPU usage (~12-25% on 8-core)
- [ ] Parallel: 8 training jobs CPU usage (should be 800%+)
- [ ] Verify no CPU bottlenecks (processes not starved)
- [ ] Compare ProcessPoolExecutor vs ThreadPoolExecutor
- [ ] Measure speedup factor (should be ~7-8× on 8 cores)

**Metrics:**
- [ ] Per-core utilization
- [ ] Total CPU percentage
- [ ] Wall-clock time vs CPU time
- [ ] Speedup factor

**Commands:**
```bash
pytest tests/performance/test_cpu_usage.py -v
```

---

### 13. ✅ Memory Usage (`tests/performance/test_memory.py`)

**Monitor memory consumption:**
- [ ] Baseline: Service startup memory (~200-300 MB)
- [ ] Per-worker memory (~100-200 MB each)
- [ ] Peak memory with 8 concurrent jobs
- [ ] Memory after 100 jobs (check for leaks)
- [ ] Verify max_tasks_per_child=10 prevents leaks
- [ ] Test PostgreSQL connection pool memory

**Metrics:**
- [ ] RSS (Resident Set Size)
- [ ] VMS (Virtual Memory Size)
- [ ] Memory growth over time
- [ ] Memory after worker recycling

**Commands:**
```bash
pytest tests/performance/test_memory.py -v
docker stats training_service --no-stream
```

---

## 🚨 Error Handling Tests

### 14. ✅ Database Connection Failures (`tests/error_handling/test_db_failures.py`)

**Test resilience:**
- [ ] PostgreSQL unavailable at startup → graceful degradation
- [ ] PostgreSQL dies during training → retry logic
- [ ] Connection pool exhaustion → queue or error
- [ ] Query timeout → proper error message
- [ ] Network partition → reconnect logic
- [ ] Invalid credentials → clear error message

**Commands:**
```bash
pytest tests/error_handling/test_db_failures.py -v
# Manual: docker compose stop postgres during tests
```

---

### 15. ✅ Process Pool Errors (`tests/error_handling/test_process_errors.py`)

**Test worker failures:**
- [ ] Worker process crashes mid-training → job marked failed
- [ ] Out of memory in worker → job marked failed
- [ ] Exception in train_model_task → error saved to DB
- [ ] Pool shutdown during active jobs → cancel_futures works
- [ ] Too many workers requested → capped at CPU_COUNT
- [ ] Worker hangs → timeout protection

**Commands:**
```bash
pytest tests/error_handling/test_process_errors.py -v
```

---

## 🔄 CI/CD Integration

### 16. ✅ Automated Test Suite (`tests/conftest.py` + GitHub Actions)

**Setup continuous testing:**
- [ ] Create `tests/conftest.py` with shared fixtures
- [ ] PostgreSQL test database fixture (create/teardown)
- [ ] FastAPI test client fixtures
- [ ] Mock data generators
- [ ] Create `.github/workflows/test.yml` for CI
- [ ] Run tests on every PR
- [ ] Generate coverage reports
- [ ] Fail PR if coverage < 80%

**Commands:**
```bash
# Run all tests
pytest tests/ -v --cov=training_service --cov=simulation_service

# Generate HTML coverage report
pytest tests/ --cov=training_service --cov=simulation_service --cov-report=html

# CI command
pytest tests/ -v --cov --cov-report=xml
```

---

## 📁 Test File Structure

```
tests/
├── conftest.py                          # Shared fixtures
├── __init__.py
│
├── unit/
│   ├── test_pg_db.py                    # Database layer tests
│   ├── test_sync_wrapper.py             # Sync wrapper tests
│   └── test_process_pool.py             # Process pool tests
│
├── api/
│   ├── test_training_api.py             # Training endpoints
│   └── test_simulation_api.py           # Simulation endpoints
│
├── integration/
│   ├── test_training_flow.py            # End-to-end training
│   └── test_simulation_flow.py          # End-to-end simulation
│
├── load/
│   ├── test_parallel_training.py        # Parallel execution
│   ├── test_concurrent_db.py            # Database concurrency
│   └── parallel_training_benchmark.py   # Standalone benchmark
│
├── validation/
│   ├── test_fingerprint.py              # Deduplication logic
│   └── test_connection_pool.py          # Pool management
│
├── performance/
│   ├── test_cpu_usage.py                # CPU utilization
│   └── test_memory.py                   # Memory consumption
│
└── error_handling/
    ├── test_db_failures.py              # Database errors
    └── test_process_errors.py           # Worker errors
```

---

## 🎯 Test Execution Plan

### Phase 1: Quick Validation (5 minutes)
```bash
python validate_migration.py              # Basic checks
python test_end_to_end.py                 # Integration test
docker compose ps                         # Services running
```

### Phase 2: Unit Tests (15 minutes)
```bash
pytest tests/unit/ -v                     # All unit tests
pytest tests/unit/test_pg_db.py -v        # Database layer
pytest tests/unit/test_sync_wrapper.py -v # Sync wrapper
```

### Phase 3: API Tests (10 minutes)
```bash
pytest tests/api/ -v                      # All API tests
```

### Phase 4: Integration Tests (20 minutes)
```bash
pytest tests/integration/ -v              # Full workflows
```

### Phase 5: Load Tests (30 minutes)
```bash
pytest tests/load/ -v                     # Parallel execution
python tests/load/parallel_training_benchmark.py
```

### Phase 6: Full Suite (1 hour)
```bash
pytest tests/ -v --cov --cov-report=html  # Everything + coverage
```

---

## 📊 Success Criteria

### Unit Tests
- ✅ 100% pass rate
- ✅ Coverage > 80% for pg_db.py, sync_wrapper.py

### API Tests
- ✅ All endpoints return expected status codes
- ✅ Response schemas validated

### Integration Tests
- ✅ End-to-end workflows complete successfully
- ✅ Data persists correctly in PostgreSQL

### Load Tests
- ✅ 8 concurrent jobs run in parallel
- ✅ Different PIDs observed
- ✅ CPU utilization > 400% (4+ cores)
- ✅ No errors under load

### Performance Tests
- ✅ Speedup factor > 6× on 8-core machine
- ✅ Memory stable after 100 jobs
- ✅ No memory leaks

### Error Handling
- ✅ Graceful degradation on DB failures
- ✅ Failed jobs marked correctly
- ✅ Clear error messages

---

## 🚀 Quick Start

### 1. Install Test Dependencies
```bash
pip install pytest pytest-asyncio pytest-cov aiohttp httpx
```

### 2. Start Services
```bash
docker compose up postgres training_service simulation_service -d
```

### 3. Run Basic Validation
```bash
python validate_migration.py
python test_end_to_end.py
```

### 4. Run Test Suite
```bash
# Create test structure
mkdir -p tests/{unit,api,integration,load,validation,performance,error_handling}

# Run tests
pytest tests/ -v
```

---

## 📝 Test Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| `training_service/pg_db.py` | 90%+ |
| `training_service/sync_db_wrapper.py` | 85%+ |
| `training_service/main.py` | 80%+ |
| `simulation_service/pg_db.py` | 90%+ |
| `simulation_service/main.py` | 80%+ |
| **Overall** | **85%+** |

---

## 🔧 Next Steps

1. **Create test directory structure**: `mkdir -p tests/{unit,api,integration,load}`
2. **Write conftest.py**: Shared fixtures for PostgreSQL, FastAPI clients
3. **Implement unit tests**: Start with `test_pg_db.py`
4. **Implement API tests**: Use FastAPI TestClient
5. **Run and iterate**: Fix failures, improve coverage
6. **Set up CI/CD**: GitHub Actions for automated testing

---

## 📚 Testing Resources

- **pytest docs**: https://docs.pytest.org/
- **pytest-asyncio**: https://github.com/pytest-dev/pytest-asyncio
- **FastAPI testing**: https://fastapi.tiangolo.com/tutorial/testing/
- **asyncpg testing**: https://magicstack.github.io/asyncpg/
- **Load testing**: https://locust.io/ or https://k6.io/

---

**Status**: Ready to implement  
**Estimated Time**: 2-3 days for full suite  
**Priority**: Start with Phase 1-2 (validation + unit tests)

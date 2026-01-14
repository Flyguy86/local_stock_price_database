# Integration Tests Quick Start Guide

## Overview

Integration tests validate complete end-to-end workflows across all system components. Unlike unit tests that test components in isolation, integration tests verify that the entire system works together correctly with real database interactions, API calls, and data flows.

## Test Coverage

### Training Service Workflows (`test_training_workflow.py`)
**Complete Workflows**:
- ✅ **Train → Save → Query Flow**: Submit job → Wait for completion → Retrieve model metadata
- ✅ **Batch Training**: Multiple models trained simultaneously
- ✅ **Parent-Child Training**: Transfer learning with model inheritance
- ✅ **Retrain Workflow**: Create new model based on existing one
- ✅ **Feature Importance**: Store and retrieve feature rankings
- ✅ **Cascade Deletion**: Verify related data deleted with model
- ✅ **Multi-Process Training**: Concurrent training jobs across CPU cores
- ✅ **Sync DB Wrapper**: Synchronous database access from worker processes

### Simulation Service Workflows (`test_simulation_workflow.py`)
**Complete Workflows**:
- ✅ **Simulate → Save → History Flow**: Run simulation → Store results → Query performance
- ✅ **Batch Simulations**: Multiple strategies tested and ranked
- ✅ **Pagination**: Large result sets with offset/limit
- ✅ **History Deletion**: Bulk cleanup of simulation records
- ✅ **Performance Metrics**: Comprehensive metric storage and retrieval
- ✅ **Configuration**: Available models and tickers
- ✅ **Strategy Comparison**: Multiple strategies on same symbol
- ✅ **Cross-Symbol Testing**: Same strategy across different symbols
- ✅ **Full Lifecycle**: Complete workflow from setup to cleanup

## Running Integration Tests

### Prerequisites

1. **PostgreSQL must be running**:
   ```bash
   docker compose up postgres -d
   ```

2. **Install test dependencies**:
   ```bash
   pip install -r tests/requirements.txt
   ```

3. **Ensure services import correctly**:
   ```bash
   python check_unit_tests.py
   ```

### Run All Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with markers
pytest tests/integration/ -m integration -v

# Run training workflow tests only
pytest tests/integration/test_training_workflow.py -v

# Run simulation workflow tests only
pytest tests/integration/test_simulation_workflow.py -v
```

### Run Specific Workflows

```bash
# Test train → save → query flow
pytest tests/integration/test_training_workflow.py::TestTrainingWorkflow::test_train_save_query_flow -v

# Test simulate → save → history flow
pytest tests/integration/test_simulation_workflow.py::TestSimulationWorkflow::test_simulate_save_history_flow -v

# Test batch training
pytest tests/integration/test_training_workflow.py::TestTrainingWorkflow::test_batch_training_workflow -v

# Test top strategies ranking
pytest tests/integration/test_simulation_workflow.py::TestSimulationWorkflow::test_batch_simulation_workflow -v
```

### Skip Slow Tests

Some integration tests are marked as "slow" (multi-process, large datasets):

```bash
# Skip slow tests
pytest tests/integration/ -m "integration and not slow" -v

# Run only slow tests
pytest tests/integration/ -m "slow" -v
```

## Test Structure

Integration tests follow end-to-end workflow patterns:

```python
class TestWorkflowName:
    """Test complete workflow."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_flow(self, db_tables):
        """
        Integration test: Step1 → Step2 → Step3 workflow.
        
        Steps:
        1. Initial action
        2. Process/transform
        3. Verify result
        4. Query/retrieve
        5. Validate state
        """
        # Step 1: Initial action (e.g., submit job)
        # ... API call or database operation
        
        # Step 2: Process (e.g., wait for completion)
        # ... async wait or status check
        
        # Step 3: Verify result
        # ... assert expected state
        
        # Step 4: Query/retrieve
        # ... fetch data via API or database
        
        # Step 5: Validate final state
        # ... comprehensive assertions
```

## Key Fixtures

- **`db_tables`**: Fresh database tables per test (auto-cleanup)
- **`sample_model_data`**: Generate mock model records
- **`test_db_pool`**: PostgreSQL connection pool for tests

Integration tests use the same fixtures as unit/API tests but combine them to test complete workflows.

## Expected Test Results

### Training Service Integration (11 tests)
- ✅ Train → Save → Query flow
- ✅ Batch training workflow
- ✅ Parent-child training
- ✅ Retrain workflow
- ✅ Feature importance workflow
- ✅ Model deletion cascade
- ✅ Concurrent training jobs (slow)
- ✅ Sync wrapper integration

### Simulation Service Integration (11 tests)
- ✅ Simulate → Save → History flow
- ✅ Batch simulation workflow
- ✅ Pagination workflow
- ✅ History deletion
- ✅ Performance metrics validation
- ✅ Configuration workflow
- ✅ Health check workflow
- ✅ Multiple strategies comparison
- ✅ Cross-symbol comparison
- ✅ Full simulation lifecycle (slow)

**Total Integration Tests**: ~22 tests  
**Expected Runtime**: 15-30 seconds (including slow tests)

## What Integration Tests Validate

### Data Flow Integrity
- ✅ Data persists correctly across all layers
- ✅ Database transactions complete successfully
- ✅ API responses match database state
- ✅ Cascade operations work correctly

### Workflow Completeness
- ✅ All steps in workflow execute in order
- ✅ Asynchronous operations complete
- ✅ Background tasks finish successfully
- ✅ Multi-step processes maintain state

### Cross-Component Interaction
- ✅ API layer talks to database layer
- ✅ Training service stores models correctly
- ✅ Simulation service retrieves models correctly
- ✅ Process pool workers access database properly

### Real-World Scenarios
- ✅ Batch operations (multiple jobs)
- ✅ Concurrent operations (parallel execution)
- ✅ Parent-child relationships (transfer learning)
- ✅ Pagination (large datasets)
- ✅ Cleanup operations (bulk deletes)

## Common Issues & Solutions

### Issue: Tests fail with "Connection refused"
**Solution**: Ensure PostgreSQL is running:
```bash
docker compose up postgres -d
sleep 3
pytest tests/integration/ -v
```

### Issue: Tests timeout
**Solution**: Some integration tests are slow. Increase timeout or skip:
```bash
pytest tests/integration/ -m "not slow" -v
```

### Issue: "Table already exists" errors
**Solution**: The `db_tables` fixture should clean up automatically. If not:
```bash
# Restart PostgreSQL
docker compose restart postgres
pytest tests/integration/ -v
```

### Issue: Intermittent failures
**Solution**: Integration tests may depend on timing. Increase wait times or run serially:
```bash
pytest tests/integration/ -v -n 1  # Serial execution
```

## Test Coverage Goals

- **Workflow Coverage**: 100% of critical user journeys
- **Data Flow**: All major CRUD operations end-to-end
- **Error Scenarios**: Cascade deletes, not found errors
- **Performance**: Batch and concurrent operations
- **Integration Points**: All service-to-database interactions

## Comparison: Unit vs API vs Integration

| Aspect | Unit Tests | API Tests | Integration Tests |
|--------|-----------|-----------|-------------------|
| **Scope** | Single function/class | Single endpoint | Complete workflow |
| **Dependencies** | Mocked/isolated | Real DB, mock data | Real DB, real flows |
| **Speed** | Fast (<1s) | Fast (1-5s) | Moderate (5-30s) |
| **Coverage** | Code logic | HTTP interface | End-to-end journey |
| **Example** | Test DB insert | Test /models endpoint | Test train→save→query |

## Performance Considerations

Integration tests are **slower** than unit/API tests because they:
- Execute complete workflows (multiple steps)
- Wait for asynchronous operations
- Test multi-process execution
- Perform multiple database round-trips

Expected runtimes:
- Fast integration test: 1-3 seconds
- Standard integration test: 3-10 seconds
- Slow integration test: 10-30 seconds (marked with `@pytest.mark.slow`)

## Integration with CI/CD

Integration tests should run in CI/CD after unit and API tests:

```yaml
# .github/workflows/test.yml
- name: Run Integration Tests
  run: |
    docker compose up postgres -d
    sleep 5
    pytest tests/integration/ -v -m "integration and not slow"
  
- name: Run Slow Integration Tests
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  run: |
    pytest tests/integration/ -v -m "slow"
```

## Debugging Integration Tests

### View detailed output
```bash
# Extra verbose
pytest tests/integration/ -vv

# Show print statements
pytest tests/integration/ -v -s

# Stop on first failure
pytest tests/integration/ -x

# Drop into debugger on failure
pytest tests/integration/ --pdb
```

### Check database state
```bash
# Connect to test database
psql -U orchestrator -d strategy_factory_test -h localhost

# List tables
\dt

# Check model records
SELECT id, algorithm, symbol, status FROM models;

# Check simulation history
SELECT strategy_id, symbol, total_return FROM simulation_history;
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pytest tests/integration/ -v` | Run all integration tests |
| `pytest tests/integration/ -k training` | Run training workflows |
| `pytest tests/integration/ -k simulation` | Run simulation workflows |
| `pytest tests/integration/ -m "not slow"` | Skip slow tests |
| `pytest tests/integration/ --collect-only` | List all integration tests |
| `pytest tests/integration/ -x --lf` | Rerun last failure and stop |

---

**Total Integration Test Count**: ~22 tests  
**Expected Pass Rate**: 100% (with PostgreSQL running)  
**Average Runtime**: 15-30 seconds (full suite)

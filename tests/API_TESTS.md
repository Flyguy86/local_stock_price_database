# API Tests Quick Start Guide

## Overview

API tests validate the HTTP endpoints of the training and simulation services. These tests ensure that all REST API endpoints work correctly, handle errors properly, and return expected data formats.

## Test Coverage

### Training Service API (`test_training_api.py`)
- **Health & Status**: `/health`, `/logs`
- **Data Endpoints**: `/data/options`, `/data/options/{symbol}`, `/data/map`, `/algorithms`
- **Model Management**: `/models`, `/models/{id}`, DELETE operations
- **Training Operations**: `/train`, `/train/batch`, `/retrain/{id}`
- **Feature Importance**: `/api/model/{id}/importance`
- **Model Config**: `/api/model/{id}/config`
- **Transfer Learning**: `/api/train_with_parent`
- **Dashboard**: `/` (HTML)

### Simulation Service API (`test_simulation_api.py`)
- **Health & Status**: `/health`, `/logs`
- **Config**: `/api/config`
- **History**: `/api/history`, `/history/top`, DELETE operations
- **Simulation**: `/api/simulate`, `/api/batch_simulate`
- **Bot Training**: `/api/train_bot`
- **Dashboard**: `/` (HTML)
- **Error Handling**: 404, 405, 422 responses

## Running API Tests

### Prerequisites

1. **PostgreSQL must be running**:
   ```bash
   docker compose up postgres -d
   ```

2. **Install test dependencies**:
   ```bash
   pip install -r tests/requirements.txt
   ```

3. **Ensure services can import**:
   ```bash
   python check_unit_tests.py
   ```

### Run All API Tests

```bash
# Run all API tests
pytest tests/api/ -v

# Run training service API tests only
pytest tests/api/test_training_api.py -v

# Run simulation service API tests only
pytest tests/api/test_simulation_api.py -v

# Run with coverage
pytest tests/api/ --cov=training_service --cov=simulation_service --cov-report=term
```

### Run Specific Test Classes

```bash
# Test health endpoints only
pytest tests/api/test_training_api.py::TestTrainingServiceHealth -v

# Test simulation history
pytest tests/api/test_simulation_api.py::TestSimulationHistory -v

# Test model management
pytest tests/api/test_training_api.py::TestModelManagement -v
```

### Run Individual Tests

```bash
# Test specific endpoint
pytest tests/api/test_training_api.py::TestTrainingServiceHealth::test_health_endpoint -v

# Test error handling
pytest tests/api/test_simulation_api.py::TestErrorHandling::test_invalid_endpoint_404 -v
```

## Test Structure

Each test file follows this pattern:

```python
class TestEndpointGroup:
    """Test related endpoints together."""
    
    @pytest.mark.asyncio
    async def test_something(self, db_tables):
        """Test description."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/endpoint")
            
        assert response.status_code == 200
        data = response.json()
        assert "expected_field" in data
```

## Key Fixtures

- **`db_tables`**: Fresh database tables for each test (auto-cleanup)
- **`sample_model_data`**: Generate mock model records
- **`api_client`**: Pre-configured httpx.AsyncClient for HTTP requests
- **`mock_data_available`**: Mock data availability to prevent file system dependencies

## Expected Test Results

### Training Service (60+ tests)
- ✅ Health endpoints (2 tests)
- ✅ Data endpoints (4 tests)
- ✅ Model management (6 tests)
- ✅ Training operations (4 tests)
- ✅ Feature importance (2 tests)
- ✅ Model config (2 tests)
- ✅ Transfer learning (1 test)
- ✅ Dashboard (1 test)

### Simulation Service (50+ tests)
- ✅ Health endpoints (2 tests)
- ✅ Config endpoints (1 test)
- ✅ History operations (6 tests)
- ✅ Simulation operations (3 tests)
- ✅ Bot training (2 tests)
- ✅ Dashboard (1 test)
- ✅ Error handling (3 tests)
- ✅ Performance metrics (1 test)

## Common Issues & Solutions

### Issue: "No file system provider found"
**Solution**: This is a workspace/terminal access issue. Use the Python test runner instead:
```bash
python -m pytest tests/api/ -v
```

### Issue: "Connection refused" or "Database not found"
**Solution**: Ensure PostgreSQL is running and test database exists:
```bash
docker compose up postgres -d
sleep 3
pytest tests/api/ -v
```

### Issue: "ModuleNotFoundError: No module named 'training_service'"
**Solution**: Ensure you're running from project root:
```bash
cd /workspaces/local_stock_price_database
pytest tests/api/ -v
```

### Issue: "asyncpg.exceptions.UndefinedTableError"
**Solution**: Tables are created automatically by fixtures. If you see this, check the `db_tables` fixture is being used:
```python
async def test_something(self, db_tables):  # ← Must include this fixture
    ...
```

### Issue: Tests pass but services fail in production
**Solution**: API tests use mock data. Run integration tests to verify with real data:
```bash
pytest tests/integration/ -v
```

## Test Coverage Goals

- **Endpoint Coverage**: 100% of all REST endpoints
- **Response Codes**: All success (200, 201) and error codes (400, 404, 405, 422, 500)
- **Request Methods**: GET, POST, DELETE for each endpoint
- **Query Parameters**: Pagination, filters, limits
- **Request Bodies**: Valid, invalid, missing fields
- **Error Scenarios**: Database failures, invalid inputs, missing resources

## Performance Considerations

API tests are **fast** because they:
- Use in-memory HTTP testing (no network overhead)
- Use test database (isolated from production)
- Don't actually train models (mock background tasks)
- Run in parallel where possible

Expected runtime: **5-10 seconds** for all API tests.

## Integration with CI/CD

These tests are designed to run in GitHub Actions:

```yaml
# .github/workflows/test.yml
- name: Run API Tests
  run: |
    docker compose up postgres -d
    sleep 5
    pytest tests/api/ -v --cov --cov-report=xml
```

## Next Steps

After API tests pass:
1. **Integration Tests**: End-to-end flows with real data
2. **Load Tests**: Concurrent requests, stress testing
3. **Performance Tests**: Response time benchmarks

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pytest tests/api/ -v` | Run all API tests |
| `pytest tests/api/ -k health` | Run tests matching "health" |
| `pytest tests/api/ --collect-only` | List all tests without running |
| `pytest tests/api/ -x` | Stop on first failure |
| `pytest tests/api/ --maxfail=3` | Stop after 3 failures |
| `pytest tests/api/ --lf` | Rerun last failed tests |
| `pytest tests/api/ -m "not slow"` | Skip slow tests |

---

**Total API Test Count**: ~110 tests  
**Expected Pass Rate**: 100% (with PostgreSQL running)  
**Average Runtime**: 5-10 seconds

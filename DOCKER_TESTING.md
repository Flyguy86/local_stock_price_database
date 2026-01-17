# Docker Compose Testing Integration

## Overview

The feature engineering pipeline includes automated end-to-end tests that run via Docker Compose. Tests validate the complete pipeline from data loading through indicator calculation, normalization, context symbols, and walk-forward fold generation.

---

## Quick Start

### Run Tests Automatically on Startup

```bash
# Build and start services with tests
docker-compose --profile test up --build
```

**What happens**:
1. All services start (api, ray_orchestrator, postgres, etc.)
2. `e2e_tests` service waits 30 seconds for Ray to initialize
3. Automated tests run (8 comprehensive tests)
4. Test results displayed in console
5. Test service exits with code 0 (pass) or 1 (fail)

---

### Run Tests Separately

```bash
# 1. Start services in background
docker-compose up -d --build

# 2. Wait for services to be ready
sleep 30

# 3. Run tests
docker-compose --profile test up e2e_tests

# 4. View test logs
docker-compose logs e2e_tests
```

---

## Test Service Configuration

**Service Name**: `e2e_tests`

**Key Settings**:
- **Image**: Same as `ray_orchestrator` (Dockerfile.ray)
- **Profile**: `test` (not started by default)
- **Restart Policy**: `no` (run once and exit)
- **Wait Time**: 30 seconds for Ray initialization
- **Exit Code**: 0 if all tests pass, 1 if any fail

**Docker Compose Excerpt**:
```yaml
e2e_tests:
  build:
    context: .
    dockerfile: Dockerfile.ray
  container_name: e2e_tests
  volumes:
    - ./data:/app/data
    - ./test_e2e_feature_pipeline.py:/app/test_e2e_feature_pipeline.py
  environment:
    RAY_ADDRESS: ray://ray_orchestrator:10001
    PYTHONPATH: /app
  command: >
    bash -c "
      sleep 30;
      python /app/test_e2e_feature_pipeline.py;
      EXIT_CODE=$$?;
      exit $$EXIT_CODE
    "
  depends_on:
    - ray_orchestrator
  restart: "no"
  profiles: ["test"]
```

---

## Test Coverage

The automated test suite includes **8 comprehensive tests**:

### ✅ Test 1: Environment Setup
- Ray cluster initialization
- Data directory availability
- Symbol availability (AAPL, QQQ, etc.)
- Import validation

### ✅ Test 2: Basic Indicator Calculation
- 25+ indicators (RSI, MACD, Bollinger Bands, etc.)
- Time features (sin/cos encoding)
- Returns (simple & log)
- Normalization ranges

### ✅ Test 3: 3-Phase Normalization Pipeline
- Phase 1: Raw calculation
- Phase 3: Simple normalization (0-100 → -1 to +1)
- Phase 4: Z-score (mean ≈ 0, std ≈ 1)

### ✅ Test 4: Context Symbol Features
- QQQ context indicators
- Relative features (close_ratio_QQQ)
- Beta calculation (rolling covariance/variance)
- VIX regime detection

### ✅ Test 5: Walk-Forward Fold Generation
- Date range validation
- Train/test split logic
- No overlap verification
- Step progression

### ✅ Test 6: Feature Engineering Version Tracking
- Version constant (v3.1)
- Preprocessor version storage
- Format validation

### ✅ Test 7: Edge Cases & Error Handling
- Empty DataFrames
- Insufficient data
- Invalid date ranges
- Duplicate timestamps

### ✅ Test 8: Performance Validation
- Processing speed (≥ 100 rows/sec)
- Memory usage (< 50 MB per symbol/day)
- Feature count (< 500 for basic config)

---

## Viewing Test Results

### Real-time Output
```bash
# Follow test logs in real-time
docker-compose --profile test up e2e_tests
```

**Expected Output**:
```
=================================================================
Waiting for Ray Orchestrator to be ready...
=================================================================

=================================================================
Starting End-to-End Feature Pipeline Tests
=================================================================

================================================================================
TEST 1: Environment Setup
================================================================================
✓ Ray initialized successfully
✓ Data directory exists: /app/data/parquet
✓ Available symbols: ['AAPL', 'QQQ', 'VIX']... (3 total)
✓ StreamingPreprocessor imported successfully

================================================================================
TEST 2: Basic Indicator Calculation
================================================================================
...

=================================================================
✓ ALL TESTS PASSED - Feature pipeline validated
=================================================================
```

---

### Post-Execution Logs
```bash
# View logs after tests complete
docker-compose logs e2e_tests

# Filter for specific test
docker-compose logs e2e_tests | grep "TEST 3"

# Check exit code
docker inspect e2e_tests --format='{{.State.ExitCode}}'
# 0 = all tests passed
# 1 = tests failed
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Feature Pipeline E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v2
      
      - name: Build Docker images
        run: docker-compose build
      
      - name: Start services
        run: docker-compose up -d
      
      - name: Wait for services to be ready
        run: sleep 45
      
      - name: Ingest test data (optional)
        run: |
          curl -X POST "http://localhost:8600/ingest/start/AAPL"
          sleep 30
      
      - name: Run E2E tests
        run: docker-compose --profile test up e2e_tests
      
      - name: Verify test results
        run: |
          docker-compose logs e2e_tests | grep "ALL TESTS PASSED"
          EXIT_CODE=$(docker inspect e2e_tests --format='{{.State.ExitCode}}')
          if [ "$EXIT_CODE" != "0" ]; then
            echo "Tests failed with exit code $EXIT_CODE"
            exit 1
          fi
      
      - name: Upload test logs (if failed)
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: test-logs
          path: |
            docker-compose logs e2e_tests
      
      - name: Cleanup
        if: always()
        run: docker-compose down -v
```

---

### GitLab CI Example

```yaml
stages:
  - build
  - test
  - cleanup

variables:
  DOCKER_DRIVER: overlay2

build:
  stage: build
  script:
    - docker-compose build

test:
  stage: test
  script:
    - docker-compose up -d
    - sleep 45
    - docker-compose --profile test up e2e_tests
    - EXIT_CODE=$(docker inspect e2e_tests --format='{{.State.ExitCode}}')
    - if [ "$EXIT_CODE" != "0" ]; then exit 1; fi
  artifacts:
    when: on_failure
    paths:
      - docker-compose logs e2e_tests

cleanup:
  stage: cleanup
  when: always
  script:
    - docker-compose down -v
```

---

## Troubleshooting

### Tests Timeout or Hang

**Issue**: Tests wait forever for Ray to initialize

**Solution**: Increase wait time in docker-compose.yml:
```yaml
command: >
  bash -c "
    sleep 60;  # Increase from 30 to 60 seconds
    python /app/test_e2e_feature_pipeline.py;
  "
```

---

### "No symbols found in data directory"

**Issue**: Tests can't find ingested data

**Solution**: Ingest data before running tests:
```bash
# Start services
docker-compose up -d

# Ingest test symbols
curl -X POST "http://localhost:8600/ingest/start/AAPL"
curl -X POST "http://localhost:8600/ingest/start/QQQ"
curl -X POST "http://localhost:8600/ingest/start/VIX"

# Wait for ingestion
sleep 60

# Run tests
docker-compose --profile test up e2e_tests
```

---

### Test Failures

**Issue**: Specific tests fail

**Debugging Steps**:
1. Check test logs:
   ```bash
   docker-compose logs e2e_tests | grep "FAILED"
   ```

2. Run tests manually for more details:
   ```bash
   docker exec -it ray_orchestrator python test_e2e_feature_pipeline.py
   ```

3. Check Ray dashboard (http://localhost:8265) for actor errors

4. Verify data availability:
   ```bash
   docker exec -it ray_orchestrator ls -lh /app/data/parquet/
   ```

---

### Container Won't Start

**Issue**: `e2e_tests` container fails to start

**Solution**: Check dependencies:
```bash
# Verify ray_orchestrator is running
docker-compose ps ray_orchestrator

# Check ray_orchestrator logs
docker-compose logs ray_orchestrator

# Manually start test container
docker-compose run --rm e2e_tests
```

---

## Manual Testing (Alternative)

If you prefer not to use Docker Compose profiles:

### Inside Container
```bash
# Exec into running container
docker exec -it ray_orchestrator bash

# Run tests manually
python test_e2e_feature_pipeline.py
```

### Local Development
```bash
# If running in dev container
cd /workspaces/local_stock_price_database
python test_e2e_feature_pipeline.py
```

---

## Best Practices

### Development Workflow
1. **Code changes**: Edit feature pipeline code
2. **Rebuild**: `docker-compose build ray_orchestrator`
3. **Test**: `docker-compose --profile test up e2e_tests`
4. **Iterate**: Fix failures and repeat

### Pre-Commit Hook
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
echo "Running E2E tests before commit..."
docker-compose up -d
sleep 30
docker-compose --profile test up e2e_tests
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "❌ Tests failed! Fix issues before committing."
  exit 1
fi

echo "✅ Tests passed!"
exit 0
```

### Nightly Build
Schedule automated tests:
```bash
# Crontab entry (runs at 2 AM daily)
0 2 * * * cd /path/to/repo && docker-compose --profile test up --build e2e_tests >> /var/log/e2e_tests.log 2>&1
```

---

## Success Criteria

**Tests pass** when:
- ✅ All 8 tests report "PASSED"
- ✅ Exit code = 0
- ✅ No error messages in logs
- ✅ Feature version == v3.1
- ✅ Performance benchmarks met

**Ready for production** when:
- ✅ Tests pass consistently (3+ runs)
- ✅ CI/CD pipeline integrated
- ✅ Data ingestion validated
- ✅ Full training pipeline tested

---

## Related Documentation

- **Test Details**: [TESTING_GUIDE.md](TESTING_GUIDE.md) - Comprehensive test documentation
- **Test Script**: [test_e2e_feature_pipeline.py](test_e2e_feature_pipeline.py) - Automated test suite
- **Features**: [README-streaming-features.md](README-streaming-features.md) - Feature engineering documentation
- **Docker Compose**: [docker-compose.yml](docker-compose.yml) - Service configuration

---

**Last Updated**: 2026-01-17 (v3.1 - Docker testing integration)

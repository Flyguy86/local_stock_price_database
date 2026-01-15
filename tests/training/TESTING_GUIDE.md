# Cohort Relationship Testing Guide

## Overview

This test suite validates the cohort vs parent/child model relationship implementation.

## Test Files

### 1. Unit Tests: `tests/training/test_cohort_relationships.py`

**Purpose**: Test core cohort logic without database dependencies

**Test Cases**:
- ✅ `test_grid_search_creates_cohort` - Grid search models share cohort_id
- ✅ `test_cohort_size_calculation` - Cohort size excludes self (N-1)
- ✅ `test_parent_child_feature_evolution` - Parent/child for feature pruning
- ✅ `test_cohort_and_parent_coexist` - Both relationships can exist
- ✅ `test_unique_fingerprints_per_cohort_sibling` - Each sibling has unique fingerprint
- ✅ `test_cohort_without_parent` - Pure grid search (no parent_model_id)
- ✅ `test_parent_without_cohort` - Pure feature evolution (no cohort_id)
- ✅ `test_multiple_cohorts_distinct` - Multiple cohorts don't overlap
- ✅ `test_fingerprint_includes_cohort_id` - Fingerprint changes with parent
- ✅ `test_find_all_siblings` - Query all models in cohort
- ✅ `test_cohort_leader_identification` - Find best model in cohort
- ✅ `test_list_all_cohorts` - List unique cohort IDs

**Run**:
```bash
python run_cohort_tests.py --unit
```

### 2. Integration Tests: `tests/training/test_cohort_integration.py`

**Purpose**: Test with real database operations

**Test Cases**:
- ✅ `test_cohort_id_column_exists` - Schema has cohort_id VARCHAR column
- ✅ `test_create_cohort_models` - Create multiple models with shared cohort_id
- ✅ `test_parent_child_without_cohort` - Parent/child relationship works
- ✅ `test_combined_cohort_and_parent` - Both fields can be set

**Requirements**:
- PostgreSQL database running
- Environment variables set (PG_HOST, PG_PORT, etc.)
- `training_service.pg_db` module available

**Run**:
```bash
# Inside Docker container
docker-compose exec training python run_cohort_tests.py --integration

# Or locally with database access
python run_cohort_tests.py --integration
```

## Running Tests

### Quick Run (All Tests)

```bash
# Make script executable
chmod +x test_cohort.sh

# Run all tests
./test_cohort.sh
```

### Individual Test Suites

```bash
# Unit tests only (no database needed)
python run_cohort_tests.py --unit

# Integration tests only (requires database)
python run_cohort_tests.py --integration

# Both (default)
python run_cohort_tests.py
```

### Using pytest

```bash
# Unit tests
pytest tests/training/test_cohort_relationships.py -v

# Integration tests (with database)
pytest tests/training/test_cohort_integration.py -v

# All tests
pytest tests/training/ -v
```

## Test Scenarios Covered

### Scenario 1: Pure Grid Search (Cohort Only)

```
Grid Search on AAPL Ridge
├─ Model 1: α=0.1, L1=0.5  (cohort_id=xyz, parent_model_id=None)
├─ Model 2: α=0.1, L1=0.7  (cohort_id=xyz, parent_model_id=None)
└─ Model 3: α=1.0, L1=0.5  (cohort_id=xyz, parent_model_id=None)

✅ Tests: test_cohort_without_parent, test_grid_search_creates_cohort
```

### Scenario 2: Pure Feature Evolution (Parent/Child Only)

```
Parent Model (100 features)  (cohort_id=None, parent_model_id=None)
  ↓
Child Model (50 features)    (cohort_id=None, parent_model_id=parent_id)

✅ Tests: test_parent_without_cohort, test_parent_child_feature_evolution
```

### Scenario 3: Combined (Grid Search on Pruned Features)

```
Original Model (100 features)
  ↓ Feature pruning
Parent Model (50 features)
  ↓ Grid search creates cohort
  ├─ Cohort Model 1 (50 features, α=0.1)  (cohort_id=xyz, parent_model_id=parent)
  └─ Cohort Model 2 (50 features, α=1.0)  (cohort_id=xyz, parent_model_id=parent)

✅ Tests: test_cohort_and_parent_coexist, test_combined_cohort_and_parent
```

## Expected Output

### Successful Unit Test Run

```
==============================================================
COHORT VS PARENT/CHILD RELATIONSHIP TESTS
==============================================================

--- Testing Cohort Relationships ---
✅ Grid search creates cohort with shared cohort_id
✅ Cohort size calculated correctly (excludes self)
✅ Parent/child relationship works for feature evolution
✅ Cohort and parent relationships coexist correctly
✅ Each cohort sibling has unique fingerprint
✅ Pure cohort works without parent_model_id
✅ Parent/child works without cohort_id
✅ Multiple cohorts remain distinct
✅ Fingerprint includes parent_model_id in hash

--- Testing Cohort Queries ---
✅ Can query all siblings in cohort
✅ Can identify best model in cohort
✅ Can list all unique cohorts

==============================================================
✅ ALL TESTS PASSED
==============================================================
```

### Successful Integration Test Run

```
==============================================================
COHORT INTEGRATION TESTS
==============================================================

Testing database schema...
✅ Database has cohort_id column

Testing cohort creation...
✅ Created cohort with 3 models, cohort_size correct

Testing parent/child without cohort...
✅ Parent/child relationship works without cohort

Testing combined relationships...
✅ Model can have both cohort_id and parent_model_id

==============================================================
✅ ALL INTEGRATION TESTS PASSED
==============================================================
```

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'training_service'
```

**Solution**: Run from project root:
```bash
cd /workspaces/local_stock_price_database
python run_cohort_tests.py
```

### Database Connection Errors

```
⚠️  Integration test dependencies not available
```

**Solution**: 
1. Start services: `docker-compose up`
2. Set environment variables
3. Run inside container: `docker-compose exec training python run_cohort_tests.py --integration`

### Migration Not Run

If integration tests fail with "cohort_id column does not exist":

```bash
# Run migration first
docker-compose exec training python training_service/migrate_cohort.py
```

## Continuous Integration

Add to CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run cohort relationship tests
  run: |
    python run_cohort_tests.py --unit
    docker-compose exec -T training python run_cohort_tests.py --integration
```

## Manual Verification

### Check Database State

```sql
-- View cohort distribution
SELECT cohort_id, COUNT(*) as siblings
FROM models
WHERE cohort_id IS NOT NULL
GROUP BY cohort_id
ORDER BY siblings DESC;

-- View combined relationships
SELECT id, symbol, algorithm, cohort_id, parent_model_id, is_grid_member
FROM models
WHERE cohort_id IS NOT NULL AND parent_model_id IS NOT NULL;
```

### Check UI

1. Navigate to training dashboard: http://localhost:8003
2. Look for cohort badges: 🔍 "N siblings ✓" or 🤝 "α=X L1=Y"
3. Click badge to open cohort modal
4. Verify all siblings displayed with hyperparameters

## Test Coverage

| Component | Test Type | Coverage |
|-----------|-----------|----------|
| Database Schema | Integration | ✅ |
| Cohort Creation | Unit + Integration | ✅ |
| Parent/Child | Unit + Integration | ✅ |
| Combined Relationships | Unit + Integration | ✅ |
| Fingerprinting | Unit | ✅ |
| Cohort Queries | Unit + Integration | ✅ |
| UI Display | Manual | ⏳ |

## Next Steps

After tests pass:

1. ✅ Run migration: `python training_service/migrate_cohort.py`
2. ✅ Restart training service
3. ✅ Train new models with grid search
4. ✅ Verify cohort display in UI
5. ✅ Check fingerprint deduplication works

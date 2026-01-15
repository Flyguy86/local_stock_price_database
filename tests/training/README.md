# Cohort Relationship Tests

## Quick Start

```bash
# Verify test files are valid
python verify_cohort_tests.py

# Run all tests
python run_cohort_tests.py

# Run specific test types
python run_cohort_tests.py --unit          # Unit tests only
python run_cohort_tests.py --integration   # Integration tests (needs DB)
```

## What's Tested

### ✅ Unit Tests (12 tests)
- Grid search creates cohort with shared `cohort_id`
- Cohort size calculation (N-1, excludes self)
- Parent/child feature evolution
- Cohort + parent relationships coexist
- Unique fingerprints per sibling
- Pure cohort (no parent)
- Pure parent/child (no cohort)
- Multiple distinct cohorts
- Fingerprint includes parent_model_id
- Query all siblings
- Find best model in cohort
- List all unique cohorts

### ✅ Integration Tests (4 tests, requires database)
- `cohort_id` column exists in schema
- Create multiple models with same cohort_id
- Parent/child without cohort works
- Combined cohort + parent works

## Test Files

| File | Purpose |
|------|---------|
| `tests/training/test_cohort_relationships.py` | Unit tests (mock DB) |
| `tests/training/test_cohort_integration.py` | Integration tests (real DB) |
| `run_cohort_tests.py` | Test runner script |
| `verify_cohort_tests.py` | Verification script |
| `tests/training/TESTING_GUIDE.md` | Detailed testing guide |

## Example Output

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

## See Also

- [COHORT_VS_PARENT_CHILD.md](../training_service/COHORT_VS_PARENT_CHILD.md) - Comprehensive documentation
- [COHORT_MIGRATION.md](../training_service/COHORT_MIGRATION.md) - Migration guide
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Detailed testing guide

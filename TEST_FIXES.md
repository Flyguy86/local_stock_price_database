# Test Fixes for Cohort Implementation

## Issues Fixed

### 1. Database Schema Compatibility (UndefinedColumnError)

**Problem**: Operations failed with `asyncpg.exceptions.UndefinedColumnError: column cohort_id does not exist`

**Cause**: The `cohort_id` column is defined in the schema initialization but may not exist in existing databases that haven't run the migration.

**Fix**: Added backward compatibility checks in `pg_db.py`:
- New method `_has_cohort_column()` checks if column exists
- `list_models()` conditionally queries `cohort_id` based on column existence
- `create_model_record()` removes `cohort_id` from INSERT if column doesn't exist
- Falls back to `NULL as cohort_id` and `0 as cohort_size` for older databases

**Code Changes**:
```python
# training_service/pg_db.py

# Helper method
async def _has_cohort_column(self, conn) -> bool:
    """Check if cohort_id column exists in models table."""
    result = await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'models' AND column_name = 'cohort_id'
        )
    """)
    return result

# In list_models()
has_cohort = await self._has_cohort_column(conn)
if has_cohort:
    # Query with cohort_id
else:
    # Fallback query

# In create_model_record()
has_cohort = await self._has_cohort_column(conn)
if not has_cohort and 'cohort_id' in data:
    data = data.copy()
    data.pop('cohort_id', None)
```

### 2. Grid Search Test Signature Mismatch

**Problem**: Test `test_elasticnet_grid_calls_save_with_training_id` checked for old positional argument pattern.

**Cause**: After cohort refactor, `_save_all_grid_models` now uses:
- `cohort_id` parameter (for grid search siblings)
- `parent_model_id` parameter (for feature evolution)
- Keyword arguments instead of all positional

**Fix**: Updated tests to check for new signature:
- `test_grid_search_parameter_signature()` now validates both `cohort_id` and `parent_model_id`
- `test_elasticnet_grid_calls_save_with_training_id()` checks for `cohort_id=training_id` pattern
- Mock calls updated to use keyword arguments

**Code Changes**:
```python
# tests/unit/test_grid_search_parent_id.py
# Before: timeframe, training_id, db, settings
# After: cohort_id=training_id, db=db, settings=settings, parent_model_id=None
```

## Migration Path

### Quick Migration (Recommended)

```bash
# Simple shell script approach
chmod +x migrate_cohort.sh
./migrate_cohort.sh
```

### For New Deployments
The schema already includes `cohort_id` in `CREATE TABLE` - no migration needed.

### For Existing Databases
Run the migration to add the column:
```bash
python training_service/migrate_cohort.py
```

Or manually:
```sql
ALTER TABLE models ADD COLUMN IF NOT EXISTS cohort_id VARCHAR;
```

## Testing

### Run Updated Tests
```bash
# All tests (should pass even without migration)
pytest tests/unit/test_pg_db.py -v
pytest tests/unit/test_grid_search_parent_id.py -v

# Cohort-specific tests
python run_cohort_tests.py --unit
```

### Expected Behavior

**Without Migration** (cohort_id column doesn't exist):
- ✅ Queries work (fallback to NULL cohort_id, 0 cohort_size)
- ✅ No errors thrown
- ⚠️ Cohort features disabled (shows as "-" in UI)

**With Migration** (cohort_id column exists):
- ✅ Queries work with cohort data
- ✅ Cohort size calculated correctly
- ✅ UI shows cohort badges and modal

## Backward Compatibility

The implementation is **fully backward compatible**:

| Scenario | Behavior |
|----------|----------|
| Fresh install | ✅ cohort_id in schema, works immediately |
| Existing DB (no migration) | ✅ Fallback query, no errors |
| After migration | ✅ Full cohort functionality |
| Old code + new DB | ✅ Works (cohort_id nullable) |
| New code + old DB | ✅ Works (fallback query) |

## Files Modified

1. **training_service/pg_db.py**
   - Added `_has_cohort_column()` method
   - Updated `list_models()` with conditional query

2. **tests/unit/test_grid_search_parent_id.py**
   - Updated signature validation test
   - Updated source code pattern check
   - Updated mock calls to use keyword args

## Verification

Check the fix is working:
```bash
# Should pass even without migration
docker-compose exec test_runner pytest tests/unit/test_pg_db.py::TestPostgreSQLDatabaseLayer::test_list_models -v

# Should pass with new signature
docker-compose exec test_runner pytest tests/unit/test_grid_search_parent_id.py -v
```

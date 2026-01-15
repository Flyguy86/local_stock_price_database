# Cohort vs Parent/Child Model Relationships

## Overview

This document clarifies the distinction between **cohort relationships** (grid search siblings) and **parent/child relationships** (feature evolution).

## Relationship Types

### 1. Cohort (Grid Search Siblings)

**Definition**: Models from the same grid search run sharing:
- Same training data
- Same feature columns
- Same reference symbols (context)
- **Different hyperparameters only**

**Database Fields**:
- `cohort_id VARCHAR` - Shared identifier for all models in the cohort
- `is_grid_member BOOLEAN` - Flags models as part of grid search
- `hyperparameters JSONB` - Specific params for each sibling

**UI Display**:
- 🔍 Badge: "N siblings ✓" (clickable to see cohort)
- 🤝 Badge: "α=X L1=Y" for cohort members
- Modal title: "🤝 Grid Search Cohort"

**Example**:
```
Cohort ID: training_abc123
├─ Model 1: Ridge α=0.1, L1=0.5
├─ Model 2: Ridge α=0.1, L1=0.7
├─ Model 3: Ridge α=1.0, L1=0.5
└─ Model 4: Ridge α=1.0, L1=0.7
```

### 2. Parent/Child (Feature Evolution)

**Definition**: Generational relationship where:
- Child inherits selected features from parent
- Parent has all features
- Child has pruned features based on parent's performance metrics

**Database Fields**:
- `parent_model_id VARCHAR` - References the model this was pruned from
- `columns_initial INT` - Original feature count
- `columns_remaining INT` - After pruning

**UI Display**:
- 👶 Badge: Shows feature reduction
- Modal: Shows feature evolution chain

**Example**:
```
Parent Model (100 features)
  ↓ Feature pruning based on importance
Child Model 1 (50 features) ← pruned low-importance features
  ↓ Further refinement
Child Model 2 (25 features) ← kept only top performers
```

## Data Model

### Schema

```sql
CREATE TABLE models (
    id VARCHAR PRIMARY KEY,
    
    -- Grid search cohort
    cohort_id VARCHAR,         -- Shared by grid siblings
    is_grid_member BOOLEAN,
    hyperparameters JSONB,
    
    -- Feature evolution lineage
    parent_model_id VARCHAR,   -- Only for pruned children
    columns_initial INT,
    columns_remaining INT,
    
    -- Other fields...
);
```

### Queries

**Find all cohort siblings**:
```sql
SELECT * FROM models 
WHERE cohort_id = 'training_abc123' 
ORDER BY hyperparameters->>'alpha';
```

**Find child models (feature evolution)**:
```sql
SELECT * FROM models 
WHERE parent_model_id = 'parent_xyz789';
```

**Cohort size**:
```sql
SELECT cohort_id, COUNT(*) as size
FROM models
WHERE cohort_id IS NOT NULL
GROUP BY cohort_id;
```

## Code Implementation

### Trainer (training_service/trainer.py)

```python
# Grid search saves cohort
_save_all_grid_models(
    grid_search, model, X_train, y_train, X_test, y_test, feature_cols_used,
    symbol, algorithm, target_col, target_transform, timeframe,
    cohort_id=training_id,  # All siblings share this
    parent_model_id=None,   # No feature evolution yet
    db=db, settings=settings, data_options=data_options
)

# Each sibling gets:
grid_record = {
    "cohort_id": cohort_id,           # Shared cohort ID
    "parent_model_id": parent_model_id,  # None for grid search
    "is_grid_member": True,
    "hyperparameters": param_set,     # Unique params
    "fingerprint": child_fingerprint,  # Unique hash including params
}
```

### Database (training_service/pg_db.py)

```python
# List query includes cohort info
query = """
    SELECT 
        m.*,
        (SELECT COUNT(*) FROM models c 
         WHERE c.cohort_id = m.cohort_id AND c.id != m.id) as cohort_size
    FROM models m
"""
```

### Frontend (training_service/static/js/dashboard.js)

```javascript
// Determine cohort display
function getGridInfo(model, allModels) {
    const cohortSize = model.cohort_size || 0;
    if (cohortSize > 0 && model.cohort_id) {
        return `🔍 ${cohortSize} siblings ✓`;
    }
    if (model.is_grid_member && model.cohort_id) {
        return `🤝 α=${hp.alpha} L1=${hp.l1_ratio}`;
    }
    return '-';
}

// Show cohort modal
async function showGridDetails(cohortId) {
    const cohortModels = Object.values(modelsCache)
        .filter(m => m.cohort_id === cohortId);
    // Display all siblings...
}
```

## Migration Path

### Existing Data

For models already trained with old parent/child structure:

1. **Grid search models** (`is_grid_member=true`):
   - Copy `parent_model_id` → `cohort_id`
   - Set `parent_model_id = NULL`
   - Add cohort leader to cohort (set its `cohort_id = id`)

2. **Feature evolution models** (not grid search):
   - Keep `parent_model_id` unchanged
   - Leave `cohort_id = NULL`

### Migration Script

Run `training_service/migrate_cohort.py` to automatically:
- Add `cohort_id` column
- Migrate grid search data
- Clear `parent_model_id` for grid members
- Update cohort leaders

See [COHORT_MIGRATION.md](COHORT_MIGRATION.md) for details.

## Future: Combining Both Relationships

A model can have BOTH cohort and parent relationships:

```
Scenario: Grid search on pruned features

Original Model (100 features)
  ↓ Feature pruning
Parent Model (50 features) ← training_def456
  ↓ Grid search creates cohort
  ├─ Cohort Model 1 (50 features, α=0.1, L1=0.5)  cohort_id=training_ghi789, parent_model_id=training_def456
  ├─ Cohort Model 2 (50 features, α=0.1, L1=0.7)  cohort_id=training_ghi789, parent_model_id=training_def456
  └─ Cohort Model 3 (50 features, α=1.0, L1=0.5)  cohort_id=training_ghi789, parent_model_id=training_def456
```

In this case:
- `cohort_id` groups the grid search siblings
- `parent_model_id` tracks the feature evolution lineage
- Both fields are set and meaningful

## Benefits

1. **Clarity**: Clear semantic distinction between exploration (cohort) and evolution (parent/child)
2. **Flexibility**: Can combine both relationships for complex scenarios
3. **Querying**: Easy to find all grid search siblings OR feature evolution children
4. **UI**: Accurate terminology ("siblings" vs "children")
5. **Fingerprinting**: Cohort models have unique fingerprints (include hyperparameters)

## Summary

| Aspect | Cohort (Grid Search) | Parent/Child (Feature Evolution) |
|--------|---------------------|----------------------------------|
| **Relationship** | Siblings | Generational |
| **Differs By** | Hyperparameters only | Feature columns |
| **Database Field** | `cohort_id` | `parent_model_id` |
| **UI Badge** | 🔍 🤝 | 👶 |
| **Use Case** | Hyperparameter tuning | Feature pruning/optimization |
| **Can Coexist?** | Yes - grid search on pruned features | Yes - grid search on pruned features |

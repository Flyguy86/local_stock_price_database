# Cohort Migration Guide

## Quick Migration

The simplest way to add the `cohort_id` column:

```bash
# Using the shell script (recommended)
chmod +x migrate_cohort.sh
./migrate_cohort.sh
```

Or manually:

```bash
# Direct SQL via docker-compose
docker-compose exec postgres psql -U postgres -d training -c "ALTER TABLE models ADD COLUMN IF NOT EXISTS cohort_id VARCHAR;"
```

## Background

Grid search models are now organized as **cohorts** (siblings with same data, different hyperparameters), distinct from **parent/child** relationships (feature evolution with pruned columns).

## Changes

- **Added**: `cohort_id VARCHAR` column - shared ID for all models in a grid search
- **Updated**: Grid search models now use `cohort_id` instead of `parent_model_id`
- **Reserved**: `parent_model_id` only for true feature evolution (pruned columns from parent)

## Running the Migration

### Option 1: Inside Docker Container

```bash
docker-compose exec training python migrate_cohort.py
```

### Option 2: Direct Connection

If you have psycopg2 installed locally:

```bash
export PG_HOST=localhost
export PG_PORT=5432
export PG_USER=postgres
export PG_PASSWORD=postgres
export PG_DATABASE=training

python training_service/migrate_cohort.py
```

### Option 3: Manual SQL

Connect to PostgreSQL and run:

```sql
-- Add column
ALTER TABLE models ADD COLUMN IF NOT EXISTS cohort_id VARCHAR;

-- Migrate grid search models
UPDATE models 
SET cohort_id = parent_model_id 
WHERE is_grid_member = true AND parent_model_id IS NOT NULL;

-- Set cohort_id for leader models
UPDATE models 
SET cohort_id = id, is_grid_member = true
WHERE id IN (
    SELECT DISTINCT parent_model_id 
    FROM models 
    WHERE is_grid_member = true AND parent_model_id IS NOT NULL
);

-- Clear parent_model_id for grid members (preserve for feature evolution only)
UPDATE models 
SET parent_model_id = NULL 
WHERE is_grid_member = true;
```

## Verification

```sql
-- Check cohort distribution
SELECT cohort_id, COUNT(*) as size
FROM models
WHERE cohort_id IS NOT NULL
GROUP BY cohort_id
ORDER BY size DESC;

-- Verify grid members have cohort_id
SELECT COUNT(*) 
FROM models 
WHERE is_grid_member = true AND cohort_id IS NOT NULL;
```

## Terminology Update

- **Before**: "Parent model with N children"
- **After**: "Cohort with N siblings"
- UI badges: 🔍 for cohort leaders, 🤝 for cohort members

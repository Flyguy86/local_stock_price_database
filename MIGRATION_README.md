# Quick Fix: Add cohort_id Column

If you see this error:
```
asyncpg.exceptions.UndefinedColumnError: column "cohort_id" of relation "models" does not exist
```

Run this simple migration:

```bash
chmod +x migrate_cohort.sh
./migrate_cohort.sh
```

That's it! The code is already backward compatible, but the migration enables full cohort functionality.

## What it does

Adds a single column to the database:
```sql
ALTER TABLE models ADD COLUMN IF NOT EXISTS cohort_id VARCHAR;
```

## Verification

Check the column was added:
```bash
docker-compose exec postgres psql -U postgres -d training -c "SELECT column_name FROM information_schema.columns WHERE table_name = 'models' AND column_name = 'cohort_id';"
```

Should output:
```
 column_name 
-------------
 cohort_id
(1 row)
```

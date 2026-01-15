#!/bin/bash
# Simple migration script to add cohort_id column

echo "Adding cohort_id column to models table..."

docker-compose exec postgres psql -U postgres -d training -c "
ALTER TABLE models ADD COLUMN IF NOT EXISTS cohort_id VARCHAR;
"

echo "Verifying column was added..."
docker-compose exec postgres psql -U postgres -d training -c "
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'models' AND column_name = 'cohort_id';
"

echo "✅ Migration complete!"
